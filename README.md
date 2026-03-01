# EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization

This repository contains the official implementation of the **EvolVE** framework. EvolVE is an evolutionary framework designed to enhance Large Language Models (LLMs) for hardware description language (Verilog) generation through advanced search strategies and rigorous evaluation.

The framework supports multiple search algorithms (including Monte Carlo Tree Search and Genetic Algorithms) and integrates with standard Verilog benchmarks like VerilogEval and RTLLM.

## Key Features
*   **Advanced Search Strategies**:
    *   **Pass@N**: Standard sampling baseline.
    *   **MCTS (Monte Carlo Tree Search)**: Tree-based search to explore the design space of Verilog implementations.
    *   **GA (Genetic Algorithm)**: A metaheuristic approach to evolve Verilog code over generations.
    *   **IGR (Idea-Guided Refinement)**: Research-paper-guided approach that generates diverse design ideas inspired by academic literature and refines implementations iteratively.
*   **Comprehensive Evaluation**:
    *   **VerilogEval**: Support for functional correctness evaluation on VerilogEval v2 and Mod-VerilogEval v2 benchmark.
    *   **RTLLM**: Support for functional correctness evaluation on RTLLM v2 benchmark.
    *   **STG (Structured Testbench Generation)**: A deterministic verification engine designed to provide high-fidelity and dense rewards.

## Directory Structure
```
EVolution/
├── src/                    # Core implementation
│   ├── search.py           # Search algorithms (MCTS, GA, Pass@N, IGR)
│   ├── igr.py              # IGR implementation with Semantic Scholar integration
│   ├── evaluator.py        # Benchmark evaluators (VerilogEval, RTLLM, STG)
│   ├── core.py             # Core classes and data structures
│   └── llm.py              # LLM interface wrappers (OpenAI compatible)
├── tool/                   # Analysis and utility scripts
│   └── analyze_report.py   # Script to summarize experiment results
├── results/                # Directory to store experiment configurations and outputs
├── benchmarks/             # Benchmark JSON files are stored here
│   ├── verilogevalv2.json          # VerilogEval v2 benchmark
│   ├── verilogevalv2_mod.json      # Mod-VerilogEval v2 benchmark
│   ├── verilogevalv2_mod_stg.json  # Mod-VerilogEval v2 benchmark with STG
│   └── rtllmv2.json                # RTLLM v2 benchmark
├── .env.example            # Example environment variable file for API keys
├── config_example.yaml     # Example configuration file for experiments
├── pyproject.toml          # Project configuration for uv
├── uv.lock                 # Dependency lock file for uv
├── main.py                 # Main experiment runner
└── run.sh                  # Helper script to launch experiments in tmux sessions
```

## Getting Started

### Prerequisites
*   Linux environment (tested on Ubuntu)
*   Python 3.11+
*   [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
*   `tmux` (for batch execution script)
*   Verilog simulation tools (e.g., iverilog) depending on the evaluator used

### Installation
1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/weiber2002/EVolution
    cd EVolution
    ```

3.  **Set up the environment:**
    ```bash
    # Install dependencies and create a virtual environment automatically
    uv sync
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the root directory to store your LLM API keys:
    ```bash
    cp .env.example .env
    # Edit .env and add your keys (OPENAI_API_KEY, GEMINI_API_KEY, etc.)
    ```
    
    **For IGR (Idea-Guided Refinement):**
    If you plan to use the IGR search strategy, you also need to configure a Semantic Scholar API key `S2_API_KEY` for research paper retrieval.
    You can obtain a free API key from [Semantic Scholar](https://www.semanticscholar.org/product/api).

## Usage

### 1. Prepare Experiment Directory
Create a dedicated folder for your experiment inside `results/` and place your benchmark configuration there.
```bash
mkdir -p results/my_experiment
cp config_example.yaml results/my_experiment/config.yaml
# Copy your benchmark dataset (e.g., verilogevalv2.json) to the experiment folder
cp benchmarks/verilogevalv2.json results/my_experiment/
```

### 2. Configure the Experiment
Edit `results/my_experiment/config.yaml` to match your desired settings.

#### Configuration Reference
A typical `config.yaml` includes the following sections:

| Section | Key | Description |
|:--------|:----|:------------|
| **output** | `folder_name` | Name of the subfolder for detailed logs/code |
| **multithreading** | `num_workers` | Number of parallel threads for evaluation |
| **llm** | `type` | LLM API type (usually `"openai"`) |
| | `base_url` | API endpoint URL (e.g., for local vLLM or Gemini) |
| | `model` | Model identifier (e.g., `gpt-4`, `gemini-2.0-flash`) |
| | `temperature` | Sampling temperature for generation |
| **evaluator** | `type` | Evaluation type:<br/>• `verilogeval` - VerilogEval benchmark (use with `verilogevalv2.json` or `verilogevalv2_mod.json`)<br/>• `stg` - STG evaluator (use with `verilogevalv2_mod_stg.json`)<br/>• `rtllm` - RTLLM benchmark (use with `rtllmv2.json`) |
| | `timeout_seconds` | Timeout for Verilog simulation execution |
| **search** | `type` | Search strategy: `pass_at_n`, `mcts`, `ga`, or `igr` |

#### Search Strategy Examples

**Pass@N (Baseline):**
```yaml
search:
  type: "pass_at_n"
  n: 100
  seed: 42
```

**Monte Carlo Tree Search (MCTS):**
```yaml
search:
  type: "mcts"
  iterations: 100
  branching_factor: 3
  c_puct: 1.4
  seed: 42
```

**Genetic Algorithm (GA):**
```yaml
search:
  type: "ga"
  iterations: 100
  num_node_selected: 2
  elite_selection_prob: 0.7
  branching_factor: 3
  num_island: 3
  population_per_island: 60
  seed: 42
```

**Idea-Guided Refinement (IGR):**
```yaml
search:
  type: "igr"
  # Number of diverse design ideas to generate
  num_ideas: 60
  # Number of implementation refinement steps per idea
  implementation_refinement_steps: 4
  seed: 42
  # Whether to run idea chains in parallel (recommended for faster execution)
  parallel_ideas: true
```

> [!NOTE]
> **IGR with Smaller LLMs:** IGR relies on structured output formats for generating search queries and scoring papers. Smaller or less capable LLMs may struggle with following these specific output format requirements, which can lead to failures in the paper collection phase. For best results, use capable models like GPT-4, Gemini, gpt-oss-120b or DeepSeek-R1 when using the IGR strategy.

### 3. Run the Experiment
Use the provided `run.sh` script to launch the experiment. This script creates a detached `tmux` session for the run, allowing long-running experiments to persist in the background.

```bash
./run.sh results/my_experiment/config.yaml
```

To view the running process:
```bash
tmux ls
tmux attach -t <session_name>
```
Or view logs and results directly in `evolution.log` and `report.csv` within the experiment folder.

### 4. Analyze Results
Once the experiment is complete, `report.csv` will be completed in the experiment folder. Use the analysis tool to generate a summary:

```bash
uv run tool/analyze_report.py results/my_experiment/report.csv
```

### 5. Understanding Search Strategy Outputs

Each search strategy produces different outputs in the `output/problem_*` directories:

| Strategy | Key Outputs | Description |
|:---------|:------------|:------------|
| **Pass@N** | `node_*.log` | Individual sampling attempts with prompts and generated code |
| **MCTS** | `node_*.log`, `relation_map.json` | Tree exploration logs and parent-child relationships |
| **GA** | `node_*.log`, `relation_map.json` | Population evolution logs and genetic lineage |
| **IGR** | `research_papers.json`, `idea_*.log`, `node_*_idea*_iter*.log`, `chain_map.json` | Research papers, design ideas, refinement chains |

### Stop the Experiment
To stop a running experiment, first identify the `tmux` session name using:
```bash
tmux ls
```
Then, terminate the session with:
```bash
tmux attach -t <session_name>
# Inside tmux, press Ctrl^C to stop the process
```