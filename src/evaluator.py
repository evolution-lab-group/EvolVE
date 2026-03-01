import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.core import Evaluator, EvaluationResult, LLM


# ============================================================================
# RTLLM Evaluator - Self-checking testbenches
# ============================================================================

class RTLLMEvaluator(Evaluator):
    """Evaluator using RTLLM-style self-checking testbenches."""
    
    def __init__(
        self,
        prob_dir: Path,
        prob_id: int,
        data: Dict[str, Any],
        log_fn: Callable[[str], None],
        log_fn_local: Callable[[str], None],
        timeout_seconds: Optional[float],
    ) -> None:
        self.prob_id = prob_id
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.timeout = timeout_seconds
        
        eval_files = data.get("eval", {})
        if not eval_files:
            raise ValueError(f"Problem {prob_id} missing 'eval' files.")
        
        # Setup working directory
        self.workdir = prob_dir / "rtllm_eval"
        self.workdir.mkdir(parents=True, exist_ok=True)
        
        # Write all evaluation files
        for filename, content in eval_files.items():
            (self.workdir / filename).write_text(content)
        
        self.tb_path = self.workdir / "tb.sv"
        if not self.tb_path.exists():
            raise FileNotFoundError(f"Problem {prob_id} missing tb.sv")
        
        # Find support files (exclude testbench and golden module)
        self.support_sources = list(self.workdir.rglob("*.sv")) + list(self.workdir.rglob("*.v"))
        self.support_sources = [
            p for p in self.support_sources
            if p != self.tb_path and "golden" not in p.name.lower() and not p.name.startswith("verified_")
        ]

    def evaluate(self, code_text: str, *, node_id: Optional[int] = None) -> EvaluationResult:
        score = 0.0
        details: Dict[str, Any] = {}
        exec_file = self.workdir / f"rtllm_{uuid.uuid4().hex}.out"
        
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sv", dir=self.workdir, delete=True
            ) as cand_fp:
                cand_fp.write(code_text)
                cand_fp.flush()
                cand_path = Path(cand_fp.name)
                
                # Compile
                compile_cmd = [
                    "iverilog", "-g2012", "-o", str(exec_file),
                    str(self.tb_path), str(cand_path), *map(str, self.support_sources)
                ]
                subprocess.run(compile_cmd, capture_output=True, text=True, 
                             timeout=self.timeout, check=True)
                
                # Simulate
                sim_proc = subprocess.run(
                    ["vvp", str(exec_file)],
                    capture_output=True, text=True, timeout=self.timeout, check=True
                )
                sim_out = sim_proc.stdout
                
                # Parse results
                match_pass = re.search(
                    r"Test completed with\s+(\d+)/(\d+)\s+failures.*pass rate =\s*([0-9.]+)%",
                    sim_out
                )
                if match_pass:
                    failures, total = int(match_pass.group(1)), int(match_pass.group(2))
                    score = float(match_pass.group(3)) / 100.0
                    details = {"failures": failures, "total_checks": total, "raw_log": sim_out}
                elif "===========Your Design Passed===========" in sim_out:
                    score = 1.0
                    details = {"failures": 0, "total_checks": None, "raw_log": sim_out}
                else:
                    details = {"error": "Unable to parse testbench output.", "raw_log": sim_out}
                
                if self.log_fn_local:
                    self.log_fn_local(f"RTLLM pass rate: {score:.2%}")
        
        except subprocess.TimeoutExpired as e:
            msg = f"[{self.prob_id}] Timed out after {e.timeout}s"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(msg)
            self.log_fn_local(msg)
            details = {
                "error": msg,
                "error_type": "TimeoutExpired",
                "timeout": e.timeout,
                "cmd": e.cmd,
                "stdout": e.stdout if e.stdout else None,
                "stderr": e.stderr if e.stderr else None,
            }
        
        except subprocess.CalledProcessError as e:
            msg = f"Failed with return code {e.returncode}"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(f"[{self.prob_id}] {msg}")
            self.log_fn_local(msg)
            details = {
                "error": msg,
                "error_type": "CalledProcessError",
                "returncode": e.returncode,
                "cmd": e.cmd,
                "stdout": e.stdout if e.stdout else None,
                "stderr": e.stderr if e.stderr else None,
            }
        
        except Exception as e:
            msg = f"{e.__class__.__name__}: {e}"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(f"[{self.prob_id}] {msg}")
            self.log_fn_local(msg)
            details = {
                "error": msg,
                "error_type": e.__class__.__name__,
            }
        
        finally:
            exec_file.unlink(missing_ok=True)
        
        return EvaluationResult(score=score, details=details)


# ============================================================================
# VerilogEval Evaluator - Built-in testbenches
# ============================================================================

class VerilogEvalEvaluator(Evaluator):
    """Evaluator using VerilogEval's built-in testbenches."""
    
    def __init__(
        self,
        prob_dir: Path,
        prob_id: int,
        data: Dict[str, Any],
        log_fn: Callable[[str], None],
        log_fn_local: Callable[[str], None],
        timeout_seconds: float,
    ) -> None:
        self.prob_id = prob_id
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.timeout = timeout_seconds
        
        # Setup testbench directory
        self.ref_path = prob_dir / "testbench"
        self.ref_path.mkdir(parents=True, exist_ok=True)
        
        eval_files = data.get("eval", {})
        if not eval_files:
            raise ValueError(f"Problem {prob_id} missing 'eval' files.")
        
        for filename, content in eval_files.items():
            (self.ref_path / filename).write_text(content)

    def evaluate(self, code_text: str, *, node_id: Optional[int] = None) -> EvaluationResult:
        score = 0.0
        details = {}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=True) as temp:
            temp.write(code_text)
            temp.flush()
            dut_path = Path(temp.name)
            
            # Find golden module and testbench
            golden_path = next(p for p in self.ref_path.rglob("*.sv") 
                             if "gold" in p.name.lower())
            tb_path = next(p for p in self.ref_path.rglob("*.sv") 
                          if "tb" in p.name.lower() or "testbench" in p.name.lower())
            
            exec_file = self.ref_path / f"a_{uuid.uuid4().hex}.out"
            
            try:
                # Compile
                subprocess.run(
                    ["iverilog", "-g2012", "-o", str(exec_file), 
                     str(tb_path), str(dut_path), str(golden_path)],
                    capture_output=True, text=True, timeout=self.timeout, check=True
                )
                
                # Simulate
                sim_out = subprocess.run(
                    ["vvp", str(exec_file)],
                    capture_output=True, text=True, timeout=self.timeout, check=True
                ).stdout
                
                # Parse results
                for pattern in [
                    r"Total mismatched samples is\s+(\d+)\s+out of\s+(\d+)\s+samples",
                    r"Mismatches:\s+(\d+)\s+in\s+(\d+)\s+samples",
                ]:
                    match = re.search(pattern, sim_out)
                    if match:
                        mismatches, total = int(match.group(1)), int(match.group(2))
                        score = (total - mismatches) / total if total > 0 else 0.0
                        break
                
                details = {"raw_log": sim_out}
                self.log_fn_local(f"Pass rate: {score:.1%}")
            
            except subprocess.TimeoutExpired as e:
                msg = f"Timed out after {e.timeout}s"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": "TimeoutExpired",
                    "timeout": e.timeout,
                    "cmd": e.cmd,
                    "stdout": e.stdout if e.stdout else None,
                    "stderr": e.stderr if e.stderr else None,
                }
                score = 0.0
            
            except subprocess.CalledProcessError as e:
                msg = f"Failed with return code {e.returncode}"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": "CalledProcessError",
                    "returncode": e.returncode,
                    "cmd": e.cmd,
                    "stdout": e.stdout if e.stdout else None,
                    "stderr": e.stderr if e.stderr else None,
                }
                score = 0.0
            
            except Exception as e:
                msg = f"Evaluation failed: {e.__class__.__name__}: {e}"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": e.__class__.__name__,
                }
                score = 0.0
            
            finally:
                exec_file.unlink(missing_ok=True)
        
        return EvaluationResult(score=score, details=details)


# ============================================================================
# STG Evaluator - Structured testbench generation
# ============================================================================

class STGEvaluator(Evaluator):
    """Evaluator using structured testbench generation (STG)."""
    
    def __init__(
        self,
        prob_dir: Path,
        prob_id: int,
        data: Dict[str, Any],
        log_fn: Callable[[str], None],
        log_fn_local: Callable[[str], None],
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.prob_id = prob_id
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.timeout = timeout_seconds or 30.0
        
        self.stg_ref_path = prob_dir / "stg"
        self.stg_ref_path.mkdir(parents=True, exist_ok=True)
        
        eval_files = data.get("eval", {})
        if not eval_files:
            raise ValueError(f"Problem {prob_id} missing 'eval' files.")
        
        for filename, content in eval_files.items():
            if filename.lower().endswith(".sv"):
                (self.stg_ref_path / filename).write_text(content)

    def evaluate(self, code_text: str, *, node_id: Optional[int] = None) -> EvaluationResult:
        score = 0.0
        details = {}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=True) as temp:
            temp.write(code_text)
            temp.flush()
            dut_path = Path(temp.name)
            
            golden_path = next(p for p in self.stg_ref_path.rglob("*.sv") 
                             if "gold" in p.name.lower())
            tb_path = next(p for p in self.stg_ref_path.rglob("*.sv") 
                          if "tb" in p.name.lower() or "testbench" in p.name.lower())
            
            exec_file = self.stg_ref_path / f"a_{uuid.uuid4().hex}.out"
            
            try:
                # Compile
                subprocess.run(
                    ["iverilog", "-g2012", "-o", str(exec_file),
                     str(tb_path), str(dut_path), str(golden_path)],
                    check=True, timeout=self.timeout, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
                # Simulate
                result = subprocess.run(
                    ["vvp", str(exec_file)],
                    capture_output=True, text=True, timeout=self.timeout
                )
                
                # Parse pass rate
                rates = re.findall(r"([0-9]+\.[0-9]+)% pass rate", result.stdout)
                if rates:
                    score = sum(float(r) for r in rates) / len(rates) / 100.0
                
                details = {"raw_log": result.stdout}
            
            except subprocess.TimeoutExpired as e:
                msg = f"STG evaluation timed out after {e.timeout}s"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": "TimeoutExpired",
                    "timeout": e.timeout,
                    "cmd": e.cmd,
                    "stdout": e.stdout if e.stdout else None,
                    "stderr": e.stderr if e.stderr else None,
                }
                score = 0.0
            
            except subprocess.CalledProcessError as e:
                msg = f"STG evaluation failed with return code {e.returncode}"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": "CalledProcessError",
                    "returncode": e.returncode,
                    "cmd": e.cmd,
                    "stdout": e.stdout if e.stdout else None,
                    "stderr": e.stderr if e.stderr else None,
                }
                score = 0.0
            
            except Exception as e:
                msg = f"STG evaluation failed: {e.__class__.__name__}: {e}"
                if node_id is not None:
                    msg += f" (node {node_id})"
                self.log_fn(f"[{self.prob_id}] {msg}")
                self.log_fn_local(msg)
                details = {
                    "error": msg,
                    "error_type": e.__class__.__name__,
                }
                score = 0.0
            
            finally:
                exec_file.unlink(missing_ok=True)
        
        return EvaluationResult(score=score, details=details)


# ============================================================================
# STGv3 Evaluator - End-to-end pipeline with LLM-generated golden model
# ============================================================================

# Prompts for STGv3 pipeline
DT_CLASS_PROMPT = """You are configuring Structured Testbench Generation (STG) for the SystemVerilog design provided below.
Classify the design into one of the allowed design types by following the guidance and classification tips:

1. `combinational`: Pure combinational logic.  
   - No clock or reset signal.  
   - Uses `always @(*)` or `assign` only.  
   - Output depends only on inputs (no stored state).

2. `seq_clocked`: Sequential design with clock.  
   - Contains `posedge` or `negedge` clock trigger (e.g., `@(posedge clk)`).  
   - May include reset and registers updated on clock edges.  
   - Has no done/valid signal.

3. `seq_done`: Sequential design with done/valid signal.  
   - Clocked sequential logic that also uses `done`, `valid`, `ready`, or `finish` to indicate completion.  
   - Often implements a multi-cycle or FSM-based operation.

Return exactly one JSON object on a single line:
{"design_type": "<combinational|seq_clocked|seq_done>"}

Do not include explanations, Markdown, or any other text.
"""

BUILD_HEADER_PROMPT = """You are acting as the golden reference model engineer for the Structured Testbench Generation (STG) flow.
Given the SPEC and the current golden-model header template, rewrite the header so it implements the same behaviour as the SPEC.

Context:
- Module name: `{module_name}`
- STG design type: `{design_type}`  // one of: combinational | seq_clocked | seq_done

=== Implementation Rules (VERY IMPORTANT) ===
1) Preserve the public interface:
   - Keep class name `GoldenModel`, public fields `clk`, `rst_n`, and output `Q` exactly as in the template.
   - You may add private internal state variables as needed (e.g., next-state, counters, FSM state).
   - Do NOT add external includes beyond those already in the header.

2) Timing semantics (how to split logic across methods):
   - `eval()` is for combinational outputs only (Mealy parts). It must NOT update sequential state.
   - `posedge_clk()` is the ONLY place that updates sequential state/regs for clocked designs (Moore state updates).
   - `negedge_clk()` is optional; leave empty unless SPEC requires falling-edge behaviour.
   - If reset is active-low (`rst_n == 0`), call `reset()` and return immediately from `posedge_clk()`.

3) Design-type tips (classification-driven placement of logic):
   - If `{design_type} == "combinational"`:
        * Implement all behaviour in `eval()` using only current inputs and current state.
        * `posedge_clk()` should NOT modify state (leave empty unless SPEC says otherwise).
   - If `{design_type} == "seq_clocked"`:
        * Implement next-state computation and state update in `posedge_clk()`.
        * `eval()` should be a no-op unless SPEC has explicit combinational outputs.
   - If `{design_type} == "seq_done"`:
        * Same as seq_clocked, but also implement completion signalling semantics (e.g., done/valid/ready) per SPEC.
        * You may add private flags and FSM state to model the handshake, but do NOT change public fields unless SPEC lists them as outputs.

4) State update atomicity:
   - When updating sequential state on `posedge_clk()`, compute using the PREVIOUS state and inputs, then assign to the registers once (use temporaries like `next_Q`).
   - Avoid partial updates that depend on already-updated fields within the same edge.

5) Types & width discipline:
   - Use `uint64_t` for 64-bit values; ensure shifts/masks stay in 64-bit.
   - Never rely on undefined behaviour. Explicitly set outputs in `reset()`.

6) Determinism & cleanliness:
   - Remove all TODO/PLACEHOLDER.
   - Keep concise comments that explain edge behaviour and state update.

=== SPEC (authoritative behaviour) ===
{SPEC}

=== DUT Source (Verilog Reference) ===
{dut_source}

=== Current golden-model header (to be rewritten) ===
```cpp
{header_source}
```
"""


class STGv3Evaluator(Evaluator):
    """Evaluator with end-to-end STG v3 pipeline including LLM-generated golden model."""
    
    def __init__(
        self,
        prob_dir: Path,
        prob_id: int,
        data: Dict[str, Any],
        log_fn: Callable[[str], None],
        log_fn_local: Callable[[str], None],
        timeout_seconds: Optional[float] = None,
        llm: Optional[LLM] = None,
    ) -> None:
        self.prob_id = prob_id
        self.log_fn = log_fn
        self.log_fn_local = log_fn_local
        self.timeout = timeout_seconds or 120.0
        self.llm = llm
        self.stg_bin = os.getenv("STG_BIN", "stg")
        
        self.stg_ref_path = prob_dir / "stgv3"
        self.stg_ref_path.mkdir(parents=True, exist_ok=True)
        
        self.spec = str(data.get("spec", ""))
        if not self.spec.strip():
            raise ValueError("STGv3 requires 'spec' in dataset.")

    def evaluate(self, code_text: str, *, node_id: Optional[int] = None) -> EvaluationResult:
        work_dir = self.stg_ref_path / f"eval_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        details = {"workdir": str(work_dir)}
        
        if not self.llm:
            return EvaluationResult(score=0.0, details={"error": "LLM not configured"})
        
        # Extract module name
        match = re.search(r"(?m)^\s*module\s+([A-Za-z_][\w$]*)", code_text)
        if not match:
            return EvaluationResult(score=0.0, details={"error": "No module name found"})
        module_name = match.group(1)
        
        try:
            # Stage 0: Classify design type
            gen = self.llm.generate(DT_CLASS_PROMPT + f"\n\nDUT:\n```verilog\n{code_text}\n```")
            design_type = self._extract_design_type(gen.content)
            if not design_type:
                raise Exception("Failed to extract design type")
            details["design_type"] = design_type
            
            # Stage 1: Generate template
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=True) as dut_fp:
                dut_fp.write(code_text)
                dut_fp.flush()
                
                out_header = work_dir / "golden_model.h"
                out_testbench = work_dir / "testbench.cpp"
                
                subprocess.run([
                    self.stg_bin, "generate",
                    "--verilog", dut_fp.name,
                    "--module", module_name,
                    "--type", design_type,
                    "--out", str(out_testbench),
                    "--out-header", str(out_header),
                    "--cc"
                ], check=True, timeout=self.timeout, capture_output=True, text=True)
                
                # Stage 2: Implement golden header via LLM
                header_text = out_header.read_text()
                prompt = BUILD_HEADER_PROMPT.format(
                    module_name=module_name,
                    design_type=design_type,
                    dut_source=code_text,
                    SPEC=self.spec,
                    header_source=header_text
                )
                gen = self.llm.generate(prompt)
                header_body = self._parse_cpp_header(gen.content)
                if not header_body:
                    raise Exception("LLM returned no header content")
                
                impl_header = work_dir / "implemented_golden_model.h"
                impl_header.write_text(header_body)
                
                # Stage 3: Compile with implemented golden model
                exe_path = work_dir / "testbench_exe"
                subprocess.run([
                    self.stg_bin, "generate",
                    "--verilog", dut_fp.name,
                    "--module", module_name,
                    "--golden", str(impl_header),
                    "--type", design_type,
                    "--out", str(out_testbench),
                    "--out-exe", str(exe_path),
                    "--cc"
                ], check=True, timeout=self.timeout, capture_output=True, text=True)
                
                # Stage 4: Execute testbench
                subprocess.run([str(exe_path)], check=True, timeout=self.timeout,
                             capture_output=True, text=True, cwd=str(work_dir))
                
                # Parse results
                stats_path = work_dir / "test_stats.json"
                score, stats = self._parse_test_stats(stats_path)
                details["test_stats"] = stats
                
                return EvaluationResult(score=score, details=details)
        
        except subprocess.TimeoutExpired as e:
            msg = f"STGv3 timed out after {e.timeout}s"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(f"[{self.prob_id}] {msg}")
            self.log_fn_local(msg)
            error_details = {
                **details,
                "error": msg,
                "error_type": "TimeoutExpired",
                "timeout": e.timeout,
                "cmd": e.cmd,
                "stdout": e.stdout if e.stdout else None,
                "stderr": e.stderr if e.stderr else None,
            }
            return EvaluationResult(score=0.0, details=error_details)
        
        except subprocess.CalledProcessError as e:
            msg = f"STGv3 failed with return code {e.returncode}"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(f"[{self.prob_id}] {msg}")
            self.log_fn_local(msg)
            error_details = {
                **details,
                "error": msg,
                "error_type": "CalledProcessError",
                "returncode": e.returncode,
                "cmd": e.cmd,
                "stdout": e.stdout if e.stdout else None,
                "stderr": e.stderr if e.stderr else None,
            }
            return EvaluationResult(score=0.0, details=error_details)
        
        except Exception as e:
            msg = f"STGv3 failed: {e.__class__.__name__}: {e}"
            if node_id is not None:
                msg += f" (node {node_id})"
            self.log_fn(f"[{self.prob_id}] {msg}")
            self.log_fn_local(msg)
            error_details = {
                **details,
                "error": msg,
                "error_type": e.__class__.__name__,
            }
            return EvaluationResult(score=0.0, details=error_details)

    def _extract_design_type(self, text: str) -> Optional[str]:
        try:
            data = json.loads(text.strip())
            dt = data.get("design_type", "").strip().lower()
            return dt if dt in {"combinational", "seq_clocked", "seq_done"} else None
        except:
            for option in ["combinational", "seq_clocked", "seq_done"]:
                if option in text.lower():
                    return option
            return None

    def _parse_cpp_header(self, text: str) -> str:
        fence = re.search(r"```(?:cpp|c\+\+|h)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return fence.group(1).strip() if fence else text.strip()

    def _parse_test_stats(self, stats_path: Path) -> tuple[float, Dict[str, Any]]:
        if not stats_path.exists():
            return 0.0, {}
        
        data = json.loads(stats_path.read_text() or "{}")
        total_tests, total_success = 0, 0
        
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            for metrics in entry.values():
                if not isinstance(metrics, dict):
                    continue
                try:
                    total_tests += max(0, int(metrics.get("tests", 0)))
                    total_success += max(0, int(metrics.get("success", 0)))
                except (TypeError, ValueError):
                    continue
        
        score = total_success / total_tests if total_tests > 0 else 0.0
        return max(0.0, min(1.0, score)), data


# ============================================================================
# Factory function
# ============================================================================

def build_evaluator(
    evaluator_cfg: Dict[str, Any],
    prob_id: int,
    prob_dir: Path,
    data: Dict[str, Any],
    log_fn: Callable[[str], None],
    log_fn_local: Callable[[str], None],
    llm: Optional[LLM] = None,
) -> Evaluator:
    """Build evaluator instance from configuration."""
    evaluator_type = str(evaluator_cfg.get("type"))
    
    evaluators = {
        "rtllm": RTLLMEvaluator,
        "verilogeval": VerilogEvalEvaluator,
        "stg": STGEvaluator,
        "stgv3": STGv3Evaluator,
    }
    
    if evaluator_type not in evaluators:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
    
    kwargs = {
        "prob_dir": prob_dir,
        "prob_id": prob_id,
        "data": data,
        "log_fn": log_fn,
        "log_fn_local": log_fn_local,
        "timeout_seconds": evaluator_cfg.get("timeout_seconds"),
    }
    
    if evaluator_type == "stgv3":
        kwargs["llm"] = llm
    
    return evaluators[evaluator_type](**kwargs)