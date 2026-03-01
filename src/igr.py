import os
import re
import time
import json
import fcntl
import requests
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core import Evaluator, EvaluationResult, LLM, Logger, SearchResult, SearchStrategy


# ============================================================================
# Semantic Scholar API Configuration
# ============================================================================

SEARCH_URL = 'https://api.semanticscholar.org/graph/v1/paper/search/'
S2_API_KEY = os.environ.get("S2_API_KEY", "")

# Global rate limiter for Semantic Scholar API
# Uses file-based locking to coordinate across multiple processes
_S2_RATE_LIMIT_FILE = "/tmp/s2_rate_limit.lock"
_S2_RATE_LIMIT_MIN_INTERVAL = 1.0  # Minimum seconds between requests


def rate_limited_request(url: str, params: dict, headers: dict, timeout: int = 10, max_retries: int = 2) -> requests.Response:
    """
    Make a rate-limited request to Semantic Scholar API.
    Ensures at most 1 request per second globally across all processes and threads.
    Uses file-based locking to coordinate between multiple running experiments.
    Retries up to max_retries times if a 429 (rate limit) status code is received.
    """
    lock_file_path = Path(_S2_RATE_LIMIT_FILE)
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    attempt = 0
    last_response = None
    
    while attempt <= max_retries:
        # Open lock file (create if doesn't exist)
        with open(lock_file_path, "a+") as lock_file:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            try:
                # Read last request timestamp from file
                lock_file.seek(0)
                content = lock_file.read().strip()
                last_request_time = float(content) if content else 0.0
                
                current_time = time.time()
                time_since_last_request = current_time - last_request_time
                
                if time_since_last_request < _S2_RATE_LIMIT_MIN_INTERVAL:
                    sleep_time = _S2_RATE_LIMIT_MIN_INTERVAL - time_since_last_request
                    time.sleep(sleep_time)
                
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                
                # Update last request time in file
                lock_file.seek(0)
                lock_file.truncate()
                lock_file.write(str(time.time()))
                lock_file.flush()
                
                last_response = response
            
            finally:
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        
        # Check if got rate limited
        if last_response and last_response.status_code == 429:
            attempt += 1
            if attempt <= max_retries:
                # Exponential backoff
                backoff_time = 2 ** attempt
                time.sleep(backoff_time)
                continue
        
        return last_response
    
    return last_response

# ============================================================================
# Prompts
# ============================================================================

INITIAL_SEARCH_PROMPT = """You are a researcher doing literature review on the topic of {TOPIC}.
You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic.

IMPORTANT: The keyword should be VERY GENERAL, not too long or specific, otherwise the search engine will fail.
Good examples of general keywords:
- "Verilog General Matrix Multiply"
- "ASICs and FPGA implementation of Image Convolutional Circuit Design"
- "Verilog Buffer Design"

Formulate your query as: KeywordQuery("keyword"). The keyword can be a concatenation of multiple keywords (just put a space between every word) but keep it general and broad to ensure good search results.

Your query (just return the query itself with no additional text):
"""

PAPER_SCORE_PROMPT = """You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:
(1) The paper is directly relevant to the topic of: {TOPIC}. Note that it should be specific to solve the problem of focus, rather than just generic methods.
(2) The paper is an empirical paper that proposes a novel method and conducts computational experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose).
(3) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects.
The papers are:
{PAPER_LIST}
Please score each paper from 1 to 10. Write the response in JSON format with "paperID: score" as the key and value for each paper.
"""

IDEA_GENERATION_PROMPT = """You are an expert hardware design researcher. Generate a creative and feasible design idea for the following specification.

SPECIFICATION:
{spec}

RELEVANT RESEARCH PAPERS (for inspiration):
{papers}

PREVIOUS IDEAS (avoid duplicating these concepts):
{prev_ideas}

The research papers above may provide useful techniques, architectural patterns, or design principles that could be adapted to this hardware design problem. Consider how these ideas might apply, but ensure your design is specific to the given specification.

Generate a novel design idea that:
1. Addresses the specification requirements
2. Is fundamentally different from previous ideas
3. Can be implemented in Verilog
4. Has clear architectural advantages
5. (Optional) Draws inspiration from the research papers if relevant

IMPORTANT: You MUST respond with EXACTLY the following format. Do not add any text before or after these sections:

### Design Idea
[In 3-5 sentences, explain: (1) the core architectural concept, (2) key innovation/differentiation, (3) main advantage. Mention paper inspiration only if directly relevant.]

### Implementation Plan
[In bullet points (5-8 items max), specify:
- Critical modules/components with their main functions
- Key design choices that impact correctness or performance
- Important data structures or signal flows
- Any non-obvious implementation details]

Do not include any introductory text, explanations, or conclusions outside these two sections. Start directly with "### Design Idea" and end after the ### Implementation Plan section.

<END>"""

IMPLEMENTATION_PROMPT = """You are implementing a hardware design based on the following idea.

SPECIFICATION:
{spec}

DESIGN IDEA:
{idea}

{previous_code}

{feedback}

Provide an improved Verilog implementation following the design idea.

Respond with exactly two sections:

### Summary
Briefly explain the design approach and any improvements made.

### Implementation
```verilog
<your Verilog code>
```

<END>"""

REFINEMENT_PROMPT = """You are refining a hardware design implementation.

SPECIFICATION:
{spec}

DESIGN IDEA:
{idea}

CURRENT IMPLEMENTATION:
```verilog
{current_code}
```

EVALUATION FEEDBACK:
Score: {score:.2%}
{feedback_details}

Analyze the issues and provide an improved implementation.

Respond with exactly two sections:

### Analysis
Explain what needs to be fixed and your improvement strategy.

### Improved Implementation
```verilog
<your improved Verilog code>
```

<END>"""


# ============================================================================
# Semantic Scholar API Functions
# ============================================================================

def search_semantic_scholar(query: str, limit: int = 20, log_fn_local=None) -> List[Dict[str, Any]]:
    """Search Semantic Scholar API with a keyword query. Rate-limited to 1 request per second."""
    if not S2_API_KEY:
        if log_fn_local:
            log_fn_local("ERROR: S2_API_KEY not found in environment variables!")
        return []
    
    query_params = {
        'query': query,
        'limit': limit,
        'fields': 'title,year,citationCount,abstract,tldr,paperId'
    }
    headers = {'x-api-key': S2_API_KEY}
    
    try:
        # Use rate-limited request function
        response = rate_limited_request(SEARCH_URL, query_params, headers, timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            papers = results.get('data', [])
            
            # Remove unwanted fields that may be included by API
            cleaned_papers = []
            for paper in papers:
                cleaned = {
                    'paperId': paper.get('paperId'),
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'citationCount': paper.get('citationCount'),
                    'abstract': paper.get('abstract'),
                    'tldr': paper.get('tldr'),
                }
                cleaned_papers.append(cleaned)
            
            return cleaned_papers
        else:
            if log_fn_local:
                log_fn_local(f"Semantic Scholar API error: {response.status_code}")
            return []
    except Exception as e:
        if log_fn_local:
            log_fn_local(f"Error searching Semantic Scholar: {e}")
        return []


def paper_filter(paper_lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out papers based on heuristics."""
    filtered = []
    for paper in paper_lst:
        # Require meaningful abstract
        abstract = paper.get("abstract", "")
        if not abstract or len(abstract.split()) < 50:
            continue
        
        # Filter out surveys/reviews
        abstract_lower = abstract.lower()
        if any(kw in abstract_lower for kw in ["survey", "review", "position paper"]):
            continue
        
        filtered.append(paper)
    return filtered


def parse_keyword_query(output: str) -> Optional[str]:
    """Parse KeywordQuery format from LLM output."""
    match = re.match(r'KeywordQuery\("([^"]+)"\)', output.strip())
    if match:
        return match.group(1)
    # Fallback: strip the function call syntax
    return output.replace("KeywordQuery(", "").replace(")", "").strip('"\'')


def format_papers_for_scoring(paper_lst: List[Dict[str, Any]]) -> str:
    """Convert papers to string for LLM scoring prompt."""
    if not paper_lst:
        return ""
    
    output = []
    for paper in paper_lst:
        lines = [f"paperId: {paper['paperId'].strip()}",
                f"title: {paper['title'].strip()}"]
        
        if paper.get("abstract"):
            lines.append(f"abstract: {paper['abstract'].strip()}")
        elif paper.get("tldr", {}).get("text"):
            lines.append(f"tldr: {paper['tldr']['text'].strip()}")
    
        if "score" in paper:
            lines.append(f"relevance score: {paper['score']}")
        
        output.append("\n".join(lines))
    
    return "\n\n".join(output)


def format_papers_for_idea_generation(papers: List[Dict[str, Any]]) -> str:
    """Format papers into a readable string for idea generation prompt."""
    if not papers:
        return "No relevant papers found."
    
    formatted = []
    for i, paper in enumerate(papers, 1):
        if not paper:
            continue
        title = paper.get('title', 'Unknown')
        year = paper.get('year', 'N/A')
        abstract = paper.get('abstract', 'No abstract available')
        citations = paper.get('citationCount', 0)
        
        formatted.append(
            f"[{i}] {title}\n"
            f"    Year: {year} | Citations: {citations}\n"
            f"    Abstract: {abstract}\n"
        )
    
    return "\n".join(formatted)


# ============================================================================
# Paper Collection Pipeline
# ============================================================================

def initial_search(topic: str, llm: LLM, log_fn_local=None) -> Tuple[str, Any]:
    """Generate initial search query using LLM.
    
    Returns:
        Tuple of (query string, generation object)
    """
    prompt = INITIAL_SEARCH_PROMPT.format(TOPIC=topic.strip())
    gen = llm.generate(prompt)
    return gen.content.strip(), gen


def score_papers(papers: List[Dict[str, Any]], topic: str, llm: LLM, log_fn_local=None) -> Tuple[Dict[str, int], Optional[Any]]:
    """Score papers using LLM.
    
    Returns:
        Tuple of (scores dict, generation object or None)
    """
    if not papers:
        return {}, None
    
    prompt = PAPER_SCORE_PROMPT.format(
        TOPIC=topic.strip(),
        PAPER_LIST=format_papers_for_scoring(papers)
    )
    
    try:
        gen = llm.generate(prompt)
        response = gen.content.strip()
        
        # Remove markdown code blocks if present
        response = re.sub(r'```(?:json)?\s*\n', '', response)
        response = re.sub(r'\n```\s*$', '', response)
        
        return json.loads(response), gen
    except Exception as e:
        if log_fn_local:
            log_fn_local(f"Error scoring papers: {e}")
        return {}, None


def collect_papers(spec: str, llm: LLM, log_fn_local=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect and score papers from Semantic Scholar.
    
    Returns:
        Tuple of (sorted papers list, metadata dict with generation objects)
    """
    paper_bank = {}
    metadata = {"queries": [], "total_papers": 0, "generations": []}
    
    # Initial search
    query_output, query_gen = initial_search(spec, llm, log_fn_local)
    query = parse_keyword_query(query_output)
    metadata["queries"].append(query)
    metadata["generations"].append(query_gen)
    
    # Execute search
    papers = search_semantic_scholar(query, limit=20, log_fn_local=log_fn_local)
    if papers:
        papers = paper_filter(papers)
        
        # Add to bank
        paper_bank = {p["paperId"]: p for p in papers}
        
        # Score papers
        scores, scoring_gen = score_papers(papers, spec, llm, log_fn_local)
        if scoring_gen:
            metadata["generations"].append(scoring_gen)
        
        # Apply scores
        for pid in paper_bank:
            paper_bank[pid]["score"] = scores.get(pid, 0)
    
    # Sort by score
    sorted_papers = sorted(
        [{'paperId': pid, **info} for pid, info in paper_bank.items()],
        key=lambda x: x.get('score', 0),
        reverse=True
    )
    
    metadata["total_papers"] = len(sorted_papers)
    
    return sorted_papers, metadata


# ============================================================================
# Utility Functions
# ============================================================================

def extract_sections(text: str) -> Dict[str, str]:
    """Extract sections from LLM response."""
    sections = {}
    
    # Extract various header sections
    for header in ["Design Idea", "Evaluation", "Analysis", "Summary"]:
        pattern = rf"###\s*{header}\s*(.*?)(?:\n###|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[header.lower().replace(" ", "_")] = match.group(1).strip()
    
    # Extract implementation plan
    plan_match = re.search(
        r"###\s*Implementation Plan\s*(.*?)(?=\n### [^#]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if plan_match:
        sections["implementation_plan"] = plan_match.group(1).strip()
    
    # Extract code
    code_match = re.search(
        r"```(?:systemverilog|verilog|sv)?\s*\n([\s\S]*?)```",
        text, re.IGNORECASE
    )
    if code_match:
        sections["code"] = code_match.group(1).strip()
    else:
        # Fallback: extract module block
        module_match = re.search(r"(module[\s\S]*?endmodule)", text, re.IGNORECASE)
        if module_match:
            sections["code"] = module_match.group(1).strip()
        
    return sections


# ============================================================================
# IGR Data Structures
# ============================================================================

@dataclass
class Idea:
    """Represents a design idea."""
    idea_id: int
    description: str
    implementation_plan: str
    refinement_history: List[str] = field(default_factory=list)


@dataclass
class RefinementNode:
    """Node in a refinement chain."""
    node_id: int
    idea: Idea
    iteration: int
    code: str
    summary: str
    evaluation: Optional[EvaluationResult] = None
    prompt: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def __post_init__(self):
        self.logger: Optional[Callable[[str], None]] = None
    
    @property
    def score(self) -> float:
        return self.evaluation.score if self.evaluation else 0.0


@dataclass
class IGRConfig:
    """Configuration for IGR search."""
    num_ideas: int
    implementation_refinement_steps: int
    seed: int
    parallel_ideas: bool = True


# ============================================================================
# IGR Search Strategy
# ============================================================================

class IGRSearch(SearchStrategy):
    """Idea-Guided Refinement search strategy."""
    
    def __init__(
        self,
        llm: LLM,
        evaluator: Evaluator,
        cfg: IGRConfig,
        search_dir: Path,
        log_fn: Optional[Callable[[str], None]] = None,
        log_fn_local: Optional[Callable[[str], None]] = None,
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
        self._idea_counter = 0
        self._llm_issue_counter: Counter[str] = Counter()
        
        search_dir.mkdir(parents=True, exist_ok=True)

    def search(self, spec: str, initial_code: str) -> SearchResult:
        """Execute IGR search with streaming idea generation and refinement."""
        # Collect papers first
        self.log_fn_local("Phase 1: Collecting research papers")
        
        papers, metadata = collect_papers(spec, self.llm, self.log_fn_local)
        self.log_fn_local(f"Collected {len(papers)} papers")
        
        # Track tokens from paper collection phase
        for gen in metadata.get("generations", []):
            if gen:
                self._track_generation(gen)
        
        # Log top papers
        if papers:
            self.log_fn_local("\nTop 10 Papers by Relevance Score:")
            for i, paper in enumerate(papers[:10], 1):
                score = paper.get('score', 0)
                title = paper.get('title', 'Unknown')
                self.log_fn_local(f"  {i}. [{score}] {title}")
        
        # Save papers
        try:
            papers_file = self.search_dir / "research_papers.json"
            with open(papers_file, 'w') as f:
                json.dump(papers, f, indent=2)
            self.log_fn_local(f"\nSaved papers to {papers_file}")
        except Exception as e:
            self.log_fn_local(f"Failed to save papers: {e}")
        
        # Phase 2: Stream idea generation and refinement
        self.log_fn_local("Phase 2: Streaming idea generation and refinement")
        
        chains = self._stream_ideas_and_refine(spec, initial_code, papers)
        self.log_fn_local(f"IGR Search completed with {len(chains)} idea chains.")
        
        if not chains:
            return SearchResult(
                code=initial_code,
                score=0.0,
                success_type="fail",
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens,
                info={"error": "Failed to generate ideas"}
            )
        
        # Find best node: highest score, then smallest iteration
        best_node = max(
            (node for chain in chains for node in chain),
            key=lambda n: (n.score, -n.iteration),
            default=None
        )
        
        if not best_node:
            return SearchResult(
                code=initial_code,
                score=0.0,
                success_type="fail",
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens,
                info={"error": "No valid implementations generated"}
            )
        
        # Determine success type
        if best_node.node_id == 0 and best_node.score >= 1.0:
            success_type = "zero-shot"
        elif best_node.score >= 1.0:
            success_type = "evolve"
        else:
            success_type = "fail"
        
        return self._build_result(best_node, chains, success_type)

    def _stream_ideas_and_refine(
        self,
        spec: str,
        initial_code: str,
        papers: List[Dict[str, Any]]
    ) -> List[List[RefinementNode]]:
        """
        Stream idea generation and immediately start refinement chains.
        Enables early stopping when any chain achieves score >= 1.0.
        """
        papers_text = format_papers_for_idea_generation(papers[:10])
        
        chains = []
        prev_ideas_text = ""
        
        # Shared state for early stopping and idea tracking
        success_found = threading.Event()
        chains_lock = threading.Lock()
        prev_ideas_list = []  # Thread-safe list to track generated ideas
        prev_ideas_lock = threading.Lock()
        
        def generate_and_refine_idea(idea_index: int):
            """Generate one idea and run its refinement chain."""
            if success_found.is_set():
                self.log_fn_local(f"Idea {idea_index}: Skipped (success found)")
                return None
            
            try:
                # Get current list of previous ideas
                with prev_ideas_lock:
                    current_prev_ideas = "\n".join(
                        f"--- Idea {i + 1} ---\n{desc}"
                        for i, desc in enumerate(prev_ideas_list)
                    )
                
                # Check again before expensive LLM call
                if success_found.is_set():
                    self.log_fn_local(f"Idea {idea_index}: Skipped before generation (success found)")
                    return None
                
                # Generate idea
                self.log_fn_local(f"Generating idea {idea_index}...")
                prompt = IDEA_GENERATION_PROMPT.format(
                    spec=spec,
                    papers=papers_text,
                    prev_ideas=current_prev_ideas
                )
                gen = self.llm.generate(prompt)
                self._track_generation(gen)
                
                # Check after LLM generation
                if success_found.is_set():
                    self.log_fn_local(f"Idea {idea_index}: Stopping after generation (success found)")
                    return None
                
                sections = extract_sections(gen.content)
                idea_desc = sections.get("design_idea", "")
                impl_plan = sections.get("implementation_plan", "")
                
                if not idea_desc:
                    self.log_fn_local(f"Idea {idea_index}: Failed to extract description")
                    return None
                
                # Create idea
                idea = Idea(
                    idea_id=self._idea_counter,
                    description=idea_desc,
                    implementation_plan=impl_plan
                )
                self._idea_counter += 1
                
                # Add to previous ideas list for future idea generation
                with prev_ideas_lock:
                    prev_ideas_list.append(idea_desc)
                
                # Log idea
                idea_log = self.search_dir / f"idea_{idea.idea_id}.log"
                Logger(idea_log).log(
                    f"Idea {idea.idea_id}\n"
                    f"Description:\n{idea.description}\n\n"
                    f"Implementation Plan:\n{idea.implementation_plan}\n"
                )
                
                # Check before starting refinement chain
                if success_found.is_set():
                    self.log_fn_local(f"Idea {idea.idea_id}: Skipping refinement (success found)")
                    return None
                
                # Start refinement chain immediately
                self.log_fn_local(f"Starting refinement chain for idea {idea.idea_id}...")
                chain = self._run_refinement_chain(
                    spec, idea, initial_code, success_found
                )
                
                # Check if this chain found success
                if chain and any(node.score >= 1.0 for node in chain):
                    self.log_fn_local(f"SUCCESS: Idea {idea.idea_id} achieved perfect score!")
                    success_found.set()
                
                return chain
                
            except Exception as e:
                self.log_fn_local(f"Error generating/refining idea {idea_index}: {e}")
                return None
        
        if self.cfg.parallel_ideas:
            # Parallel mode: Generate ideas and run refinement chains in parallel
            with ThreadPoolExecutor(max_workers=self.cfg.num_ideas) as executor:
                futures = []
                
                for i in range(self.cfg.num_ideas):
                    future = executor.submit(generate_and_refine_idea, i)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        chain = future.result()
                        if chain:
                            with chains_lock:
                                chains.append(chain)
                        
                        # Immediate return on success
                        if success_found.is_set():
                            # Cancel all remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            self.log_fn_local(f"Success found - returning immediately with {len(chains)} chain(s)")
                            
                            return chains
                            
                    except Exception as e:
                        self.log_fn_local(f"Chain execution failed: {e}")
        else:
            # Sequential mode: Generate ideas one by one and run refinement
            for i in range(self.cfg.num_ideas):
                chain = generate_and_refine_idea(i)
                if chain:
                    chains.append(chain)
                
                # Immediate return on success
                if success_found.is_set():
                    self.log_fn_local(f"Success found - returning immediately with {len(chains)} chain(s)")
                    return chains
        
        self.log_fn_local(f"\nCompleted {len(chains)} refinement chains")
        return chains

    def _run_refinement_chain(
        self,
        spec: str,
        idea: Idea,
        initial_code: str,
        success_found: threading.Event
    ) -> List[RefinementNode]:
        """
        Run a single refinement chain for an idea with early stopping support.
        
        Args:
            spec: Problem specification
            idea: Design idea to implement
            initial_code: Initial/baseline code
            success_found: Event that signals when any chain has found success
        """
        chain = []
        
        # Check early stop before starting
        if success_found.is_set():
            self.log_fn_local(f"Idea {idea.idea_id} - Skipped (success already found)")
            return chain
        
        # Initial implementation
        node = self._initial_implementation(spec, idea, initial_code, success_found)
        if not node:
            return chain
        
        chain.append(node)
        self.log_fn_local(f"Idea {idea.idea_id} - Initial: score={node.score:.3f}")
        
        if node.score >= 1.0:
            success_found.set()
            return chain
        
        # Iterative refinement
        for iteration in range(1, self.cfg.implementation_refinement_steps + 1):
            # Check early stop before each iteration
            if success_found.is_set():
                self.log_fn_local(
                    f"Idea {idea.idea_id} - Stopping at iter {iteration} (success found elsewhere)"
                )
                break
            
            refined_node = self._refine_implementation(spec, idea, chain[-1], iteration, success_found)
            
            if not refined_node:
                break
            
            chain.append(refined_node)
            self.log_fn_local(f"Idea {idea.idea_id} - Iter {iteration}: score={refined_node.score:.3f}")
            
            if refined_node.score >= 1.0:
                success_found.set()
                break
        
        return chain

    def _initial_implementation(
        self,
        spec: str,
        idea: Idea,
        initial_code: str,
        success_found: Optional[threading.Event] = None
    ) -> Optional[RefinementNode]:
        """Generate initial implementation from idea."""
        try:
            # Check if success already found
            if success_found and success_found.is_set():
                self.log_fn_local(f"Idea {idea.idea_id}: Skipping implementation (success found)")
                return None
            
            prev_code_section = ""
            if initial_code.strip():
                prev_code_section = f"\nBASELINE CODE (for reference):\n```verilog\n{initial_code}\n```\n"
            
            prompt = IMPLEMENTATION_PROMPT.format(
                spec=spec,
                idea=f"{idea.description}\n\n{idea.implementation_plan}",
                previous_code=prev_code_section,
                feedback=""
            )
            
            gen = self.llm.generate(prompt)
            self._track_generation(gen)
            
            # Check again after LLM generation
            if success_found and success_found.is_set():
                self.log_fn_local(f"Idea {idea.idea_id}: Stopping after code generation (success found)")
                return None
            
            sections = extract_sections(gen.content)
            code = sections.get("code", "")
            summary = sections.get("summary", "")
            
            if not code:
                self.log_fn_local(f"Idea {idea.idea_id}: No code generated")
                return None
            
            node = self._create_node(
                idea=idea,
                iteration=0,
                code=code,
                summary=summary,
                prompt=prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens
            )
            
            self._evaluate_node(node)
            return node
            
        except Exception as e:
            self.log_fn_local(f"Error in initial implementation for idea {idea.idea_id}: {e}")
            return None

    def _refine_implementation(
        self,
        spec: str,
        idea: Idea,
        parent_node: RefinementNode,
        iteration: int,
        success_found: Optional[threading.Event] = None
    ) -> Optional[RefinementNode]:
        """Refine implementation based on feedback."""
        try:
            # Check if success already found
            if success_found and success_found.is_set():
                self.log_fn_local(f"Idea {idea.idea_id} - Iter {iteration}: Skipping (success found)")
                return None
            
            feedback_details = json.dumps(
                parent_node.evaluation.details, indent=2
            ) if parent_node.evaluation else "No evaluation details"
            
            prompt = REFINEMENT_PROMPT.format(
                spec=spec,
                idea=f"{idea.description}\n\n{idea.implementation_plan}",
                current_code=parent_node.code,
                score=parent_node.score,
                feedback_details=feedback_details
            )
            
            gen = self.llm.generate(prompt)
            self._track_generation(gen)
            
            # Check again after LLM generation
            if success_found and success_found.is_set():
                self.log_fn_local(f"Idea {idea.idea_id} - Iter {iteration}: Stopping after generation (success found)")
                return None
            
            sections = extract_sections(gen.content)
            code = sections.get("code", "")
            analysis = sections.get("analysis", "")
            
            if not code:
                self.log_fn_local(f"Idea {idea.idea_id} - Iter {iteration}: No code generated")
                return None
            
            node = self._create_node(
                idea=idea,
                iteration=iteration,
                code=code,
                summary=analysis,
                prompt=prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens
            )
            
            self._evaluate_node(node)
            return node
            
        except Exception as e:
            self.log_fn_local(f"Error refining idea {idea.idea_id} iteration {iteration}: {e}")
            return None

    def _create_node(self, idea: Idea, iteration: int, code: str, summary: str, **kwargs) -> RefinementNode:
        """Create a refinement node."""
        node = RefinementNode(
            node_id=self._node_counter,
            idea=idea,
            iteration=iteration,
            code=code,
            summary=summary,
            **kwargs
        )
        self._node_counter += 1
        
        node.logger = Logger(
            self.search_dir / f"node_{node.node_id}_idea{idea.idea_id}_iter{iteration}.log"
        ).log
        
        # Build the log message with prompt if available
        log_message = (
            f"Node {node.node_id}\n"
            f"Idea ID: {idea.idea_id} | Iteration: {iteration}\n"
        )
        
        if node.prompt:
            log_message += f"\nPrompt:\n{node.prompt}\n\n{'-' * 60}\n\n"
        
        log_message += (
            f"Summary:\n{summary}\n\n"
            f"Code:\n```verilog\n{code}\n```\n"
            f"Tokens - input: {node.prompt_tokens}, output: {node.completion_tokens}\n"
            f"{'-' * 60}"
        )
        
        node.logger(log_message)
        
        return node

    def _evaluate_node(self, node: RefinementNode) -> None:
        """Evaluate a node."""
        try:
            evaluation = self.evaluator.evaluate(node.code, node_id=node.node_id)
            node.evaluation = evaluation
            
            if node.logger:
                node.logger(
                    f"Evaluation Result\n"
                    f"Score: {evaluation.score:.3f}\n"
                    f"Details: {json.dumps(evaluation.details, indent=2)}\n"
                    f"{'-' * 60}"
                )
        
        except Exception as e:
            self.log_fn_local(f"Evaluation failed for node {node.node_id}: {e}")
            node.evaluation = EvaluationResult(score=0.0, details={"error": str(e)})

    def _track_generation(self, generation) -> None:
        """Track token usage and issues."""
        self.total_input_tokens += generation.prompt_tokens
        self.total_output_tokens += generation.completion_tokens
        reason = generation.finish_reason or "unknown"
        if reason.lower() != "stop":
            self._llm_issue_counter[reason] += 1

    def _build_result(
        self,
        best_node: RefinementNode,
        chains: List[List[RefinementNode]],
        success_type: str
    ) -> SearchResult:
        """Build final search result."""
        # Build chain map
        chain_map = {}
        for chain_idx, chain in enumerate(chains):
            if not chain:
                continue
            
            chain_map[f"chain_{chain_idx}"] = {
                "idea_id": chain[0].idea.idea_id,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "iteration": node.iteration,
                        "score": node.score,
                    }
                    for node in chain
                ],
                "best_score": max(node.score for node in chain),
            }
        
        # Save chain map
        try:
            (self.search_dir / "chain_map.json").write_text(
                json.dumps(chain_map, indent=2)
            )
        except Exception as e:
            self.log_fn_local(f"Failed to save chain map: {e}")
                
        return SearchResult(
            code=best_node.code,
            score=best_node.score,
            success_type=success_type,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            info={
                "best_node": best_node,
                "num_ideas": len(chains),
                "iterations": best_node.node_id + 1,
                "chain_map": chain_map,
                "llm_issues": dict(self._llm_issue_counter),
            }
        )