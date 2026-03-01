import argparse
import csv
import json
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from dotenv import load_dotenv
load_dotenv()

from src.core import Logger, SearchResult
from src.evaluator import build_evaluator
from src.llm import build_llm
from src.search import build_search



def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping.")
    
    return data


def load_dataset(base_dir: Path) -> List[Dict[str, Any]]:
    """Load dataset JSON file (excluding output_*.json files)."""
    candidates = [
        p for p in sorted(base_dir.glob("*.json"))
        if not p.name.startswith("output_")
    ]
    
    if not candidates:
        raise FileNotFoundError("Dataset file not found.")
    
    with candidates[0].open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Validate dataset
    seen_ids = set()
    for idx, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {idx} must be a mapping.")
        if "i" not in entry:
            raise KeyError(f"Dataset entry {idx} missing key 'i'.")
        if entry["i"] in seen_ids:
            raise ValueError(f"Duplicate problem ID {entry['i']}.")
        seen_ids.add(entry["i"])
    
    return dataset


class ProgressTracker:
    """Thread-safe progress tracking for multiple problems."""
    
    def __init__(self, prob_ids: List[int], log_fn: Callable[[str], None], csv_path: Optional[Path] = None):
        self.log_fn = log_fn
        self.lock = threading.Lock()
        self.csv_path = csv_path
        self.status: Dict[int, Dict[str, Any]] = {
            pid: {"state": "pending"} for pid in prob_ids
        }
    
    def update(self, prob_id: int, state: str, **info: Any) -> None:
        """Update status for a problem."""
        now = time.time()
        
        with self.lock:
            current = dict(self.status.get(prob_id, {}))
            
            if state == "running" and "started_at" not in current:
                current["started_at"] = now
            
            if state in {"completed", "failed"}:
                current.setdefault("started_at", now)
                info.setdefault("finished_at", now)
                start_time = current.get("started_at", now)
                info.setdefault("duration", info["finished_at"] - start_time)
            
            current.update(info)
            current["state"] = state
            self.status[prob_id] = current
        
        # Write CSV immediately when a problem finishes (completed or failed)
        if state in {"completed", "failed"} and self.csv_path is not None:
            try:
                self.write_csv(self.csv_path)
            except Exception as exc:
                self.log_fn(f"Failed to update CSV after problem {prob_id} {state}: {exc}")
    
    def summarize(self, reason: str, *, final: bool = False) -> None:
        """Log current progress summary."""
        with self.lock:
            groups: Dict[str, List[int]] = {
                "pending": [],
                "running": [],
                "completed": [],
                "failed": [],
            }
            success_breakdown: Counter[str] = Counter()
            
            for prob_id, data in self.status.items():
                state = data.get("state", "pending")
                groups.setdefault(state, []).append(prob_id)
                
                if state == "completed":
                    success_breakdown[data.get("success_type", "unknown")] += 1
        
        parts = [
            f"{label}({len(items)}): {','.join(map(str, sorted(items))) or '-'}"
            for label, items in groups.items()
        ]
        
        message = f"[Progress] {reason} | " + " | ".join(parts)
        
        if final and success_breakdown:
            breakdown = ", ".join(
                f"{key}:{value}" for key, value in sorted(success_breakdown.items())
            )
            message += f" | success_types={breakdown}"
        
        self.log_fn(message)
    
    def write_report(self, path: Path) -> None:
        """Write human-readable progress report."""
        with self.lock:
            snapshot = {pid: dict(data) for pid, data in self.status.items()}
        
        headers = [
            "Problem", "State", "Success", "Score", "Duration(s)",
            "Started", "Finished", "LLM Issues"
        ]
        
        rows = []
        for prob_id in sorted(snapshot.keys()):
            data = snapshot[prob_id]
            
            # Format fields
            score = data.get("score")
            score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
            
            duration = data.get("duration")
            duration_text = f"{float(duration):.2f}" if isinstance(duration, (int, float)) else "-"
            
            issues = data.get("llm_issue_counts", {})
            issues_text = ", ".join(
                f"{k}:{v}" for k, v in sorted(issues.items()) if isinstance(v, int) and v > 0
            ) or "None"
            
            rows.append([
                str(prob_id),
                str(data.get("state", "pending")),
                str(data.get("success_type") or "-"),
                score_text,
                duration_text,
                self._format_time(data.get("started_at")),
                self._format_time(data.get("finished_at")),
                issues_text,
            ])
        
        # Format table
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]
        
        def format_row(row):
            return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        
        lines = [
            format_row(headers),
            "-+-".join("-" * w for w in widths),
            *[format_row(row) for row in rows]
        ]
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    def write_csv(self, path: Path) -> None:
        """Write CSV metrics report."""
        with self.lock:
            snapshot = {pid: dict(data) for pid, data in self.status.items()}
        
        headers = [
            "problem_id", "final_state", "score", "iterations",
            "total_nodes_generated", "total_input_tokens", "total_output_tokens"
        ]
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for prob_id in sorted(snapshot.keys()):
                data = snapshot[prob_id]
                
                # Calculate total nodes if possible
                iterations = data.get("iterations")
                branching = data.get("branching_factor")
                total_nodes = data.get("total_nodes_generated")
                
                if total_nodes is None and isinstance(iterations, int) and isinstance(branching, int):
                    total_nodes = iterations * branching
                
                writer.writerow([
                    prob_id,
                    data.get("final_state") or data.get("success_type") or data.get("state"),
                    data.get("score"),
                    iterations,
                    total_nodes,
                    data.get("total_input_tokens"),
                    data.get("total_output_tokens"),
                ])
    
    @staticmethod
    def _format_time(value: Optional[float]) -> str:
        """Format timestamp."""
        if not value:
            return "-"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def run_problem(
    prob_id: int,
    data: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    llm,
    tracker: ProgressTracker,
    global_log_fn: Callable[[str], None],
) -> Dict[str, Any]:
    """Run evolution for a single problem."""
    tracker.update(prob_id, "running", llm_issue_counts={})
    global_log_fn(f"[{prob_id}] Starting evolution...")
    
    prob_dir = output_dir / f"problem_{prob_id}"
    prob_dir.mkdir(parents=True, exist_ok=True)
    local_log_fn = Logger(prob_dir / f"problem_{prob_id}.log").log
    
    try:
        # Build evaluator
        evaluator = build_evaluator(
            evaluator_cfg=config.get("evaluator", {}),
            prob_id=prob_id,
            prob_dir=prob_dir,
            data=data,
            log_fn=global_log_fn,
            log_fn_local=local_log_fn,
            llm=llm,
        )
        
        # Build search strategy
        search = build_search(
            search_cfg=config.get("search", {}),
            prob_dir=prob_dir,
            llm=llm,
            evaluator=evaluator,
            log_fn=global_log_fn,
            log_fn_local=local_log_fn,
        )
        
        # Run search
        global_log_fn(f"[{prob_id}] Running {search.__class__.__name__}")
        result: SearchResult = search.search(
            spec=data["spec"],
            initial_code=data.get("initial_code", ""),
        )
        
        # Extract metrics
        iterations = result.info.get("iterations") if isinstance(result.info, dict) else None
        branching = getattr(getattr(search, "cfg", None), "branching_factor", None)
        
        # For IGR
        if branching is None and hasattr(search, "cfg") and hasattr(search.cfg, "num_ideas"):
            branching = 1
        
        total_nodes = iterations * branching if iterations and branching else None
        
        llm_issues = {}
        if isinstance(result.info, dict):
            issues = result.info.get("llm_issues", {})
            if isinstance(issues, dict):
                llm_issues = {str(k): int(v) for k, v in issues.items() if int(v) > 0}
        
        # Update tracker
        tracker.update(
            prob_id,
            "completed",
            score=result.score,
            success_type=result.success_type,
            final_state=result.success_type,
            iterations=iterations,
            total_input_tokens=result.total_input_tokens,
            total_output_tokens=result.total_output_tokens,
            branching_factor=branching,
            total_nodes_generated=total_nodes,
            llm_issue_counts=llm_issues,
        )
        
        global_log_fn(
            f"[{prob_id}] Completed: score={result.score:.3f} "
            f"success_type={result.success_type} "
            f"tokens={result.total_input_tokens}/{result.total_output_tokens}"
        )
        
        return {
            "code": result.code,
            "score": result.score,
            "success_type": result.success_type,
        }
    
    except Exception as exc:
        tracker.update(prob_id, "failed", error=str(exc), final_state="fail")
        global_log_fn(f"[{prob_id}] Failed: {exc}")
        raise


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hardware design evolution framework.")
    parser.add_argument("-c", "--config", required=True, help="Path to configuration YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Load config and exit.")
    args = parser.parse_args(argv)
    
    # Load configuration
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base_dir = config_path.parent
    
    # Setup output directory
    output_dir = base_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(base_dir)
    problems = [(entry["i"], entry) for entry in dataset]
    prob_ids = [pid for pid, _ in problems]
    
    # Setup logging
    log_path = base_dir / "evolution.log"
    log_path.unlink(missing_ok=True)
    log_fn = Logger(log_path).log
    
    # Setup CSV path
    csv_path = base_dir / "report.csv"
    
    # Setup progress tracking
    tracker = ProgressTracker(prob_ids, log_fn, csv_path)
    tracker.summarize("initial snapshot")
    
    # Build LLM
    log_fn("Building LLM...")
    llm = build_llm(config.get("llm", {}))
    
    # Determine parallelism
    num_workers = int(config.get("multithreading", {}).get("num_workers", 1) or 1)
    
    # Track best programs
    best_programs: Dict[int, Dict[str, Any]] = {}
    best_programs_lock = threading.Lock()
    
    def run_with_tracking(prob_id: int, data: Dict[str, Any]) -> None:
        """Wrapper to track best program."""
        try:
            result = run_problem(prob_id, data, config, output_dir, llm, tracker, log_fn)
            with best_programs_lock:
                best_programs[prob_id] = result
        except Exception as e:
            log_fn(f"[{prob_id}] Exception in run_with_tracking: {e}")
            with best_programs_lock:
                best_programs.setdefault(prob_id, {
                    "code": "",
                    "score": None,
                    "success_type": "error"
                })
    
    # Run problems
    try:
        if num_workers <= 1 or len(prob_ids) <= 1:
            # Sequential execution
            for prob_id, data in problems:
                run_with_tracking(prob_id, data)
                # Log progress after each problem in sequential mode
                tracker.summarize("sequential progress update")
        else:
            # Parallel execution
            max_workers = min(num_workers, len(prob_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_with_tracking, prob_id, data): prob_id
                    for prob_id, data in problems
                }
                completed_count = 0
                for future in as_completed(futures):
                    prob_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log_fn(f"[{prob_id}] Unhandled exception in future: {exc}")
                    
                    # Log progress periodically in parallel mode
                    completed_count += 1
                    if completed_count % max(1, len(prob_ids) // 10) == 0 or completed_count == len(prob_ids):
                        tracker.summarize(f"parallel progress ({completed_count}/{len(prob_ids)} completed)")
    
    finally:
        # Final reporting
        tracker.summarize("final snapshot", final=True)
        
        report_path = base_dir / "report.txt"
        try:
            tracker.write_report(report_path)
            log_fn(f"Report saved to {report_path}")
        except Exception as exc:
            log_fn(f"Failed to write report: {exc}")
        
        csv_path = base_dir / "report.csv"
        try:
            tracker.write_csv(csv_path)
            log_fn(f"CSV saved to {csv_path}")
        except Exception as exc:
            log_fn(f"Failed to write CSV: {exc}")
        
        # Write output dataset
        try:
            with best_programs_lock:
                best_snapshot = dict(best_programs)
            
            enriched_dataset = []
            for entry in dataset:
                new_entry = dict(entry)
                prob_id = new_entry.get("i")
                best_entry = best_snapshot.get(prob_id, {})
                new_entry["best_program"] = best_entry.get("code", "")
                enriched_dataset.append(new_entry)
            
            dataset_path = next(p for p in base_dir.glob("*.json") if not p.name.startswith("output_"))
            output_dataset_path = base_dir / f"output_{dataset_path.name}"
            
            with output_dataset_path.open("w", encoding="utf-8") as f:
                json.dump(enriched_dataset, f, ensure_ascii=False, indent=2)
            
            log_fn(f"Output dataset saved to {output_dataset_path}")
        
        except Exception as exc:
            log_fn(f"Failed to write output dataset: {exc}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())