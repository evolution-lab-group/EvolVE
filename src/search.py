import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.core import Evaluator, EvaluationResult, LLM, Logger, SearchResult, SearchStrategy
from src.igr import IGRConfig, IGRSearch


def extract_code_and_summary(text: Optional[str]) -> Tuple[str, str]:
    """Parse LLM response into code and summary sections."""
    if not text or not isinstance(text, str):
        return "", ""
    
    text = text.strip()
    code, summary = "", ""
    
    try:
        # Extract summary
        summary_match = re.search(
            r"###\s*Summary\s*(.*?)(?:\n###|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(0).strip()
        
        # Extract code from fence or module block
        code_match = re.search(
            r"```(?:systemverilog|verilog|sv)?\s*\n([\s\S]*?)```",
            text, re.IGNORECASE
        )
        if code_match:
            code = code_match.group(1).strip()
        else:
            fallback = re.search(r"(module[\s\S]*?endmodule)", text, re.IGNORECASE)
            if fallback:
                code = fallback.group(1).strip()
    except re.error:
        pass
    
    return code, summary


# ============================================================================
# Pass@N Search - Simple sampling strategy
# ============================================================================

@dataclass
class PassAtNConfig:
    attempts: int


class PassAtNSearch(SearchStrategy):
    """Simple sampling strategy that tries N independent generations."""
    
    def __init__(
        self,
        llm: LLM,
        evaluator: Evaluator,
        cfg: PassAtNConfig,
        search_dir: Path,
        log_fn: Optional[Callable[[str], None]] = None,
        log_fn_local: Optional[Callable[[str], None]] = None,
    ) -> None:
        if cfg.attempts <= 0:
            raise ValueError("Pass@N requires n > 0")
        
        self.llm = llm
        self.evaluator = evaluator
        self.cfg = cfg
        self.search_dir = search_dir
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._llm_issue_counter: Counter[str] = Counter()
        
        search_dir.mkdir(parents=True, exist_ok=True)

    def search(self, spec: str, initial_code: str) -> SearchResult:
        best_code = initial_code
        best_score = 0.0
        best_eval: Optional[EvaluationResult] = None
        success_type = "fail"
        
        # Evaluate initial code if provided
        if initial_code.strip():
            try:
                baseline_eval = self.evaluator.evaluate(initial_code, node_id=0)
                best_eval = baseline_eval
                best_score = baseline_eval.score
                self.log_fn_local(f"Baseline score: {baseline_eval.score:.3f}")
                
                if best_score >= 1.0:
                    return SearchResult(
                        code=initial_code,
                        score=best_score,
                        success_type="zero-shot",
                        total_input_tokens=0,
                        total_output_tokens=0,
                        info={"iterations": 0, "evaluation": baseline_eval}
                    )
            except Exception as exc:
                self.log_fn_local(f"Baseline evaluation failed: {exc}")
        
        # Run N attempts
        for attempt in range(self.cfg.attempts):
            prompt = f"{spec}\n\nProvide SystemVerilog module:\n```verilog\n<code>\n```"
            generation = self.llm.generate(prompt)
            self._track_generation(generation)
            
            code, summary = extract_code_and_summary(generation.content)
            
            # Log attempt
            attempt_logger = Logger(self.search_dir / f"attempt_{attempt}.log").log
            attempt_logger(
                f"Attempt {attempt}\n"
                f"Code:\n{code or '<no code>'}\n"
                f"Tokens: {generation.prompt_tokens} in, {generation.completion_tokens} out"
            )
            
            # Evaluate
            try:
                evaluation = self.evaluator.evaluate(code, node_id=attempt + 1)
                self.log_fn_local(f"Attempt {attempt} score: {evaluation.score:.3f}")
                attempt_logger(f"Score: {evaluation.score}")
                
                if evaluation.score > best_score or best_eval is None:
                    best_score = evaluation.score
                    best_code = code
                    best_eval = evaluation
                
                if evaluation.score >= 1.0:
                    success_type = "sample"
                    break
            
            except Exception as exc:
                self.log_fn_local(f"Attempt {attempt} evaluation failed: {exc}")
                continue
        
        return SearchResult(
            code=best_code,
            score=best_score,
            success_type=success_type,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            info={"iterations": attempt + 1, "evaluation": best_eval}
        )

    def _track_generation(self, generation) -> None:
        self.total_input_tokens += generation.prompt_tokens
        self.total_output_tokens += generation.completion_tokens
        reason = generation.finish_reason or "unknown"
        if reason.lower() != "stop":
            self._llm_issue_counter[reason] += 1


# ============================================================================
# MCTS - Monte Carlo Tree Search
# ============================================================================

def uct(q: float, n: int, n_parent: int, c_puct: float) -> float:
    """Upper Confidence Bound for Trees (modified version)."""
    if n == 0:
        return float("inf")
    return q + c_puct * math.sqrt(max(1, n_parent)) / (1 + n)


def uct_original(q: float, n: int, n_parent: int, c_puct: float) -> float:
    """Original UCT formula."""
    if n == 0:
        return float("inf")
    return q / n + c_puct * math.sqrt(math.log(max(1, n_parent)) / n)


@dataclass
class MCTSConfig:
    iterations: int
    branching_factor: int
    c_puct: float
    seed: int
    uct_type: str = "mod"  # "mod" or "original"


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    node_id: int
    code_text: str
    depth: int = 0
    value_sum: float = 0.0
    visits: int = 0
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    summary: Optional[str] = None
    prompt: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def __post_init__(self):
        self.logger: Optional[Callable[[str], None]] = None
        self._log_initialization()
    
    def _log_initialization(self):
        """Log complete node initialization details."""
        if self.logger:
            self.logger(
                f"Initialized node {self.node_id} at depth {self.depth}.\n"
                f"Code:\n{self.code_text}\n"
                f"Summary:\n{self.summary or '<no summary>'}\n"
                f"Prompt:\n{self.prompt or '<no prompt>'}\n"
                f"Token usage - input: {self.prompt_tokens}, output: {self.completion_tokens}\n"
                f"-------------------------"
            )
    
    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0
    
    @property
    def score(self) -> float:
        return self.evaluation.score if self.evaluation else 0.0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def best_child(self, c_puct: float, uct_type: str = "mod") -> "MCTSNode":
        if not self.children:
            raise ValueError("Leaf node has no children")
        
        n_parent = max(1, self.visits)
        uct_fn = uct_original if uct_type == "original" else uct
        
        scored = [
            (uct_fn(ch.value_sum if uct_type == "original" else ch.q_value,
                   ch.visits, n_parent, c_puct), ch)
            for ch in self.children
        ]
        return max(scored, key=lambda x: x[0])[1]


class MCTSSearch(SearchStrategy):
    """Monte Carlo Tree Search for hardware design optimization."""
    
    def __init__(
        self,
        llm: LLM,
        evaluator: Evaluator,
        cfg: MCTSConfig,
        search_dir: Path,
        log_fn: Optional[Callable[[str], None]],
        log_fn_local: Optional[Callable[[str], None]],
    ) -> None:
        self.llm = llm
        self.evaluator = evaluator
        self.cfg = cfg
        self.search_dir = search_dir
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._node_counter = 0
        self._llm_issue_counter: Counter[str] = Counter()
        
        search_dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)

    def search(self, spec: str, initial_code: str) -> SearchResult:
        # Generate initial code if not provided
        if not initial_code.strip():
            generation = self.llm.generate(self._build_prompt(spec))
            self._track_generation(generation)
            initial_code, initial_summary = extract_code_and_summary(generation.content)
        else:
            initial_summary = ""
        
        # Create root node
        root = self._create_node(code=initial_code, summary=initial_summary, depth=0)
        self._evaluate_node(root)
        
        # Check for zero-shot success
        if root.score >= 1.0:
            return self._build_result(root, "zero-shot", 0)
        
        best_node = root
        
        # MCTS iterations
        for iteration in range(self.cfg.iterations):
            # Selection
            leaf = self._select(root)
            
            # Expansion
            children = self._expand(leaf, spec)
            
            # Evaluation
            candidates = children or [leaf]
            unevaluated = [n for n in candidates if n.visits == 0]
            if unevaluated:
                self._evaluate_nodes(unevaluated)
            
            # Track best
            for node in candidates:
                if node.score > best_node.score:
                    best_node = node
            
            self.log_fn_local(
                f"[MCTS iter {iteration + 1}] root Q={root.q_value:.3f} "
                f"best={best_node.score:.3f} visits={root.visits}"
            )
            
            if best_node.score >= 1.0:
                self.log_fn_local(f"Early stop at iteration {iteration + 1}")
                break
        
        success_type = "evolve" if best_node.score >= 1.0 else "fail"
        return self._build_result(best_node, success_type, iteration + 1)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCT."""
        cur = node
        while not cur.is_leaf():
            cur = cur.best_child(self.cfg.c_puct, self.cfg.uct_type)
        return cur

    def _expand(self, parent: MCTSNode, spec: str) -> List[MCTSNode]:
        """Expand node by generating children."""
        prompts = [
            self._build_prompt(
                spec,
                parent.code_text,
                parent.summary,
                parent.evaluation.details if parent.evaluation else {}
            )
            for _ in range(self.cfg.branching_factor)
        ]
        
        try:
            with ThreadPoolExecutor(max_workers=self.cfg.branching_factor) as executor:
                generations = list(executor.map(self.llm.generate, prompts))
        except Exception as exc:
            self.log_fn_local(f"Expansion failed: {exc}")
            return []
        
        children = []
        for prompt, gen in zip(prompts, generations):
            self._track_generation(gen)
            code, summary = extract_code_and_summary(gen.content)
            child = self._create_node(
                code=code,
                summary=summary,
                depth=parent.depth + 1,
                parent=parent,
                prompt=prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens
            )
            parent.children.append(child)
            children.append(child)
        
        return children

    def _evaluate_node(self, node: MCTSNode) -> None:
        """Evaluate and backpropagate."""
        evaluation = self.evaluator.evaluate(node.code_text, node_id=node.node_id)
        node.evaluation = evaluation
        node.value_sum += evaluation.score
        node.visits += 1
        
        if node.logger:
            node.logger(
                f"Applied evaluation to node {node.node_id}.\n"
                f"Score: {evaluation.score}\n"
                f"Details: {evaluation.details}\n"
                f"-------------------------"
            )
        
        self._backpropagate(node, evaluation.score)

    def _evaluate_nodes(self, nodes: Iterable[MCTSNode]) -> None:
        """Evaluate multiple nodes in parallel."""
        nodes = list(nodes)
        with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            future_map = {
                executor.submit(self.evaluator.evaluate, n.code_text, node_id=n.node_id): n
                for n in nodes
            }
            for future in as_completed(future_map):
                node = future_map[future]
                try:
                    evaluation = future.result()
                    node.evaluation = evaluation
                    node.value_sum += evaluation.score
                    node.visits += 1
                    
                    if node.logger:
                        node.logger(
                            f"Applied evaluation to node {node.node_id}.\n"
                            f"Score: {evaluation.score}\n"
                            f"Details: {evaluation.details}\n"
                            f"-------------------------"
                        )
                    
                    self._backpropagate(node, evaluation.score)
                except Exception as exc:
                    self.log_fn_local(f"Evaluation failed for node {node.node_id}: {exc}")

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        cur = node.parent
        while cur is not None:
            cur.value_sum += reward
            cur.visits += 1
            cur = cur.parent

    def _create_node(self, code: str, depth: int, **kwargs) -> MCTSNode:
        """Create a new MCTS node with logger."""
        node = MCTSNode(
            node_id=self._node_counter,
            code_text=code,
            depth=depth,
            **kwargs
        )
        self._node_counter += 1
        node.logger = Logger(self.search_dir / f"node_{node.node_id}.log").log
        node.logger(
            f"Created node {node.node_id} at depth {depth}.\n"
            f"Code:\n{node.code_text}\n"
            f"Summary:\n{node.summary or '<no summary>'}\n"
            f"Prompt:\n{node.prompt or '<no prompt>'}\n"
            f"Token usage - input: {node.prompt_tokens}, output: {node.completion_tokens}\n"
            f"-------------------------"
        )
        return node

    def _build_prompt(
        self,
        spec: str,
        code: str = None,
        summary: str = None,
        eval_details: Dict[str, Any] = None
    ) -> str:
        """Build prompt for LLM generation."""
        parts = ["You are improving hardware designs. Produce a summary and improved SystemVerilog module."]
        parts.append(f"SPEC:\n{spec}")
        
        if summary:
            parts.append(f"PREVIOUS SUMMARY:\n{summary}\n")
        if eval_details:
            parts.append(f"PREVIOUS EVAL:\n{eval_details}\n")
        if code:
            parts.append(f"PREVIOUS CODE:\n```verilog\n{code}\n```")
        
        parts.append(
            "Respond with exactly two sections:\n"
            "### Summary\n"
            "- Briefly explain the design trade-offs.\n\n"
            "### Improved Code\n"
            "```verilog\n"
            "<your code here>\n"
            "```\n"
            "<END>"
        )
        
        return "\n".join(parts)

    def _track_generation(self, generation) -> None:
        """Track token usage and issues."""
        self.total_input_tokens += generation.prompt_tokens
        self.total_output_tokens += generation.completion_tokens
        reason = generation.finish_reason or "unknown"
        if reason.lower() != "stop":
            self._llm_issue_counter[reason] += 1

    def _build_result(self, node: MCTSNode, success_type: str, iterations: int) -> SearchResult:
        """Build final search result."""
        # Build relationship map
        relation_map = {}
        stack = [node]
        visited = set()
        
        # Walk back to root and gather all nodes
        cur = node
        while cur.parent:
            stack.append(cur.parent)
            cur = cur.parent
        root = cur
        
        # Now traverse from root to build complete map
        stack = [root]
        while stack:
            current = stack.pop()
            if current.node_id in visited:
                continue
            visited.add(current.node_id)
            relation_map[current.node_id] = {
                "parent": current.parent.node_id if current.parent else None,
                "children": [ch.node_id for ch in current.children],
                "score": current.score,
                "visits": current.visits,
            }
            stack.extend(current.children)
        
        # Save relation map
        try:
            (self.search_dir / "relation_map.json").write_text(
                json.dumps(relation_map, indent=2)
            )
        except Exception as exc:
            self.log_fn_local(f"Failed to save relation map: {exc}")
        
        return SearchResult(
            code=node.code_text,
            score=node.score,
            success_type=success_type,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            info={
                "node": node,
                "iterations": iterations,
                "relation_map": relation_map,
                "llm_issues": dict(self._llm_issue_counter),
            }
        )


# ============================================================================
# Genetic Algorithm
# ============================================================================

@dataclass
class GAConfig:
    iterations: int
    branching_factor: int
    population_per_island: int
    num_node_selected: int
    elite_selection_prob: float
    elite_size: int
    num_island: int
    seed: int


@dataclass
class GANode:
    """Node in the genetic algorithm population."""
    node_id: int
    code_text: str
    summary: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    prompt: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parents: List["GANode"] = field(default_factory=list)
    children: List["GANode"] = field(default_factory=list)
    
    def __post_init__(self):
        self.logger: Optional[Callable[[str], None]] = None
    
    @property
    def score(self) -> float:
        return self.evaluation.score if self.evaluation else 0.0


class GASearch(SearchStrategy):
    """Genetic algorithm with island model for hardware design optimization."""
    
    def __init__(
        self,
        llm: LLM,
        evaluator: Evaluator,
        cfg: GAConfig,
        search_dir: Path,
        log_fn: Optional[Callable[[str], None]] = None,
        log_fn_local: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.llm = llm
        self.evaluator = evaluator
        self.cfg = cfg
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.search_dir = search_dir
        self._node_counter = 0
        self._population: List[List[GANode]] = [[] for _ in range(cfg.num_island)]
        self._all_nodes: Dict[int, GANode] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._llm_issue_counter: Counter[str] = Counter()
        
        search_dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)

    def search(self, spec: str, initial_code: str) -> SearchResult:
        # Generate initial program if needed
        if not initial_code.strip():
            generation = self.llm.generate(spec)
            self._track_generation(generation)
            initial_code, initial_summary = extract_code_and_summary(generation.content)
        else:
            initial_summary = ""
        
        # Create and evaluate initial node
        initial_node = self._create_node(code=initial_code, summary=initial_summary)
        initial_eval = self.evaluator.evaluate(initial_node.code_text, node_id=initial_node.node_id)
        initial_node.evaluation = initial_eval
        
        # Check for zero-shot success
        if initial_eval.score >= 1.0:
            return self._build_result(initial_node, "zero-shot", 0)
        
        # Initialize first island with initial node
        self._population[0].append(initial_node)
        best_node = initial_node
        current_island = 0
        
        # Evolution loop
        for iteration in range(self.cfg.iterations):
            island_pop = self._population[current_island]
            
            # Refill empty island with best node
            if not island_pop and best_node:
                island_pop.append(best_node)
            
            # Selection
            selected = self._selection(current_island)
            if not selected:
                current_island = (current_island + 1) % self.cfg.num_island
                continue
            
            # Crossover/Mutation
            children = self._expand(selected, spec)
            if not children:
                current_island = (current_island + 1) % self.cfg.num_island
                continue
            
            # Evaluation
            unevaluated = [n for n in children if n.evaluation is None]
            if unevaluated:
                self._evaluate_nodes(unevaluated)
            
            # Update best
            for node in children:
                if node.evaluation and node.score > best_node.score:
                    best_node = node
            
            self.log_fn_local(
                f"[GA iter {iteration + 1}] island={current_island} best={best_node.score:.3f}"
            )
            
            # Survival selection
            self._populate(current_island, children)
            
            if best_node.score >= 1.0:
                self.log_fn_local(f"Early stop at iteration {iteration + 1}")
                break
            
            current_island = (current_island + 1) % self.cfg.num_island
        
        success_type = "evolve" if best_node.score >= 1.0 else "fail"
        return self._build_result(best_node, success_type, iteration + 1)

    def _selection(self, island_index: int) -> List[GANode]:
        """Select nodes from island for reproduction."""
        island = self._population[island_index]
        if not island:
            return []
        
        target_count = min(len(island), self.cfg.num_node_selected)
        selected = []
        available = list(island)
        
        for _ in range(target_count):
            if not available:
                break
            
            if random.random() < self.cfg.elite_selection_prob:
                # Elite selection: pick from top nodes
                sorted_nodes = sorted(available, key=lambda n: n.score, reverse=True)
                elite_size = min(self.cfg.elite_size, len(sorted_nodes))
                choice = random.choice(sorted_nodes[:elite_size])
            else:
                # Random selection
                choice = random.choice(available)
            
            selected.append(choice)
            available.remove(choice)
        
        return selected

    def _expand(self, parents: List[GANode], spec: str) -> List[GANode]:
        """Generate children from selected parents."""
        if not parents:
            return []
        
        prompts = [self._build_prompt(spec, parents) for _ in range(self.cfg.branching_factor)]
        
        try:
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                generations = list(executor.map(self.llm.generate, prompts))
        except Exception as exc:
            self.log_fn_local(f"Expansion failed: {exc}")
            return []
        
        children = []
        for prompt, gen in zip(prompts, generations):
            self._track_generation(gen)
            code, summary = extract_code_and_summary(gen.content)
            child = self._create_node(
                code=code,
                summary=summary,
                prompt=prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                parents=list(parents)
            )
            
            for parent in parents:
                parent.children.append(child)
            
            children.append(child)
        
        return children

    def _populate(self, island_index: int, children: List[GANode]) -> None:
        """Add children to island and enforce population limit."""
        island = self._population[island_index]
        island.extend(children)
        
        # Enforce population limit
        if len(island) > self.cfg.population_per_island > 0:
            island.sort(key=lambda n: n.score, reverse=True)
            del island[self.cfg.population_per_island:]

    def _evaluate_nodes(self, nodes: Iterable[GANode]) -> None:
        """Evaluate multiple nodes in parallel."""
        nodes = list(nodes)
        with ThreadPoolExecutor(max_workers=min(len(nodes), self.cfg.branching_factor)) as executor:
            future_map = {
                executor.submit(self.evaluator.evaluate, n.code_text, node_id=n.node_id): n
                for n in nodes
            }
            for future in as_completed(future_map):
                node = future_map[future]
                try:
                    node.evaluation = future.result()
                    if node.logger:
                        node.logger(
                            f"Applied evaluation to node {node.node_id}.\n"
                            f"Score: {node.evaluation.score}\n"
                            f"Details: {node.evaluation.details}\n"
                            f"-------------------------"
                        )
                except Exception as exc:
                    self.log_fn_local(f"Evaluation failed for node {node.node_id}: {exc}")

    def _create_node(self, code: str, **kwargs) -> GANode:
        """Create a new GA node with logger."""
        node = GANode(node_id=self._node_counter, code_text=code, **kwargs)
        self._node_counter += 1
        self._all_nodes[node.node_id] = node
        node.logger = Logger(self.search_dir / f"node_{node.node_id}.log").log
        summary_text = (node.summary or "").strip() or "<no summary>"
        prompt_text = (node.prompt or "").strip() or "<no prompt>"
        code_text = (node.code_text or "").strip() or "<no code>"
        node.logger(
            f"Initialized node {node.node_id}.\n"
            f"Summary:\n{summary_text}\n"
            f"Prompt:\n{prompt_text}\n"
            f"Token usage - input: {node.prompt_tokens}, output: {node.completion_tokens}\n"
            f"Code:\n```verilog\n{code_text}\n```\n"
            f"-------------------------"
        )
        
        return node

    def _build_prompt(self, spec: str, parents: List[GANode]) -> str:
        """Build crossover prompt from parent nodes."""
        parts = [
            "You are improving hardware designs. Combine parent strengths to produce improved SystemVerilog.\n",
            f"SPEC:\n{spec}\n"
        ]
        
        if parents:
            parts.append("PARENTS:")
            for i, parent in enumerate(parents, 1):
                parts.append(f"\n### Parent {i} (score: {parent.score:.3f})")
                if parent.summary:
                    parts.append(f"Summary: {parent.summary}")
                parts.append(f"Code:\n```verilog\n{parent.code_text}\n```")
        
        parts.append(
            "\nRespond with:\n"
            "### Summary\n<explain improvements>\n\n"
            "### Improved Code\n```verilog\n<code>\n```"
        )
        
        return "\n".join(parts)

    def _track_generation(self, generation) -> None:
        """Track token usage and issues."""
        self.total_input_tokens += generation.prompt_tokens
        self.total_output_tokens += generation.completion_tokens
        reason = generation.finish_reason or "unknown"
        if reason.lower() != "stop":
            self._llm_issue_counter[reason] += 1

    def _build_result(self, node: GANode, success_type: str, iterations: int) -> SearchResult:
        """Build final search result."""
        # Build relationship map
        relation_map = {
            node_id: {
                "parents": [p.node_id for p in n.parents],
                "children": [c.node_id for c in n.children],
                "score": n.score,
            }
            for node_id, n in self._all_nodes.items()
        }
        
        # Save relation map
        try:
            (self.search_dir / "relation_map.json").write_text(
                json.dumps(relation_map, indent=2)
            )
        except Exception as exc:
            self.log_fn_local(f"Failed to save relation map: {exc}")
        
        return SearchResult(
            code=node.code_text,
            score=node.score,
            success_type=success_type,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            info={
                "node": node,
                "iterations": iterations,
                "relation_map": relation_map,
                "llm_issues": dict(self._llm_issue_counter),
            }
        )


# ============================================================================
# Factory function
# ============================================================================

def build_search(
    search_cfg: Dict[str, Any],
    prob_dir: Path,
    llm: LLM,
    evaluator: Evaluator,
    log_fn: Callable[[str], None],
    log_fn_local: Callable[[str], None],
) -> SearchStrategy:
    """Build search strategy from configuration."""
    search_type = str(search_cfg.get("type"))
    
    if search_type == "pass_at_n":
        cfg = PassAtNConfig(attempts=int(search_cfg.get("n")))
        return PassAtNSearch(llm, evaluator, cfg, prob_dir / "search", log_fn, log_fn_local)
    elif search_type == "mcts":
        cfg = MCTSConfig(
            iterations=int(search_cfg.get("iterations")),
            branching_factor=int(search_cfg.get("branching_factor")),
            c_puct=float(search_cfg.get("c_puct")),
            seed=int(search_cfg.get("seed")),
            uct_type=str(search_cfg.get("uct_type", "mod")).lower()
        )
        return MCTSSearch(llm, evaluator, cfg, prob_dir / "search", log_fn, log_fn_local)
    elif search_type == "ga":
        cfg = GAConfig(
            iterations=int(search_cfg.get("iterations")),
            branching_factor=int(search_cfg.get("branching_factor")),
            population_per_island=int(search_cfg.get("population_per_island")),
            num_node_selected=int(search_cfg.get("num_node_selected")),
            elite_selection_prob=float(search_cfg.get("elite_selection_prob")),
            elite_size=int(search_cfg.get("elite_size", 1)),
            num_island=int(search_cfg.get("num_island")),
            seed=int(search_cfg.get("seed")),
        )
        return GASearch(llm, evaluator, cfg, prob_dir / "search", log_fn, log_fn_local)
    elif search_type == "igr":        
        cfg = IGRConfig(
            num_ideas=int(search_cfg.get("num_ideas", 5)),
            implementation_refinement_steps=int(search_cfg.get("implementation_refinement_steps", 5)),
            seed=int(search_cfg.get("seed", 42)),
            parallel_ideas=bool(search_cfg.get("parallel_ideas", True)),
        )
        return IGRSearch(llm, evaluator, cfg, prob_dir / "search", log_fn, log_fn_local)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")