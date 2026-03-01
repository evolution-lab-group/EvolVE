from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol


@dataclass
class EvaluationResult:
    """Result from evaluating a hardware design."""
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMGeneration:
    """Container for a single LLM response with token usage tracking."""
    raw_out: Any
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Optional[str] = None


@dataclass
class SearchResult:
    """Result from running a search strategy."""
    code: str
    score: float
    success_type: str  # "zero-shot", "evolve", "sample", "fail"
    total_input_tokens: int
    total_output_tokens: int
    info: Dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    """Protocol for evaluating hardware designs."""
    def evaluate(self, code_text: str, *, node_id: Optional[int] = None) -> EvaluationResult:
        ...


class LLM(Protocol):
    """Protocol for language model inference."""
    def generate(self, prompt: str) -> LLMGeneration:
        ...


class SearchStrategy(Protocol):
    """Protocol for search/optimization strategies."""
    def search(
        self,
        spec: str,
        initial_code: str,
    ) -> SearchResult:
        ...


class Logger:
    """Simple file logger with timestamps."""
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        import time
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass