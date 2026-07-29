# SEER: Sequence Entropy–Energy Gating for Selective Prediction and Text-to-SQL Refinement

SEER is a training-free inference-time framework for uncertainty-aware text-to-SQL generation. It augments a frozen instruction-tuned code model with a parameter-free gate based on either sequence-level Shannon entropy or logit energy.

For each query, SEER generates a complete SQL candidate, measures uncertainty over the generated token sequence, and optionally executes the candidate against the target SQLite database. If the candidate fails to execute or remains above the calibrated uncertainty threshold, SEER appends feedback to the original prompt and generates another candidate. The process stops when the candidate executes successfully and satisfies the uncertainty gate, or when the maximum number of refinement steps is reached.

The implementation supports:

- Qwen2.5-Coder-7B-Instruct
- DeepSeek-Coder-6.7B-Instruct
- entropy-gated and energy-gated refinement
- execution-guided repair using SQLite errors
- first-position, sequence-mean, and sequence-maximum uncertainty analysis
- post-hoc execution accuracy, validity, convergence, AUROC, and accuracy-coverage evaluation

## Method overview

Given a database schema, external evidence, and a natural-language question, SEER performs the following steps:

1. **Generate** a complete SQL candidate using greedy decoding.
2. **Measure uncertainty** using sequence-level Shannon entropy or logit energy.
3. **Execute** the candidate against the target SQLite database.
4. **Validate** the candidate:
   - accept it when execution succeeds and the gate value is at or below the calibrated threshold;
   - otherwise append either the SQLite error or a fixed self-check cue and refine again.
5. **Apply the compute cap** after `K_MAX` iterations.

In the default execution-guided setting, candidate selection follows the implementation exactly:

- executable candidates always outrank non-executable candidates;
- among candidates with the same executability status, the lower gate value wins;
- if no candidate executes, SEER returns the non-executable candidate with the lowest gate value;
- exact ties retain the earliest generated candidate.

Evaluation metrics are computed only after inference and are not available to the model or uncertainty gate.

## Requirements

- Python 3.10+
- CUDA-capable GPU
- approximately 16 GB GPU memory when using 4-bit NF4 quantization
- BIRD-SQL development split
- access to the Hugging Face model repositories when authentication is required

The default configuration uses:

- 4-bit NF4 quantization
- bfloat16 compute
- greedy decoding
- maximum prompt length of 4096 tokens
- maximum generation length of 256 tokens
- repetition penalty of 1.1
- `K_MAX = 5`
- sequence-mean gating
- execution-guided refinement

## Installation

```bash
git clone https://github.com/leonardo-alexander/SEER.git
cd SEER

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Set the required environment variables:

```bash
export BIRD_ROOT=/path/to/unzipped/bird
export HF_TOKEN=your_huggingface_token   # only when required
```

On Windows PowerShell:

```powershell
$env:BIRD_ROOT="D:\path\to\bird"
$env:HF_TOKEN="your_huggingface_token"
```

A template is provided in `.env.example`.

## Expected BIRD-SQL layout

The loader searches several common BIRD-SQL layouts automatically. The recommended structure is:

```text
$BIRD_ROOT/
└── dev/
    ├── dev.json
    └── dev_databases/
        └── <db_id>/
            └── <db_id>.sqlite
```

Only the development split is required. Threshold calibration uses the built-in synthetic SQL set in `seer_runner.py`, so BIRD training data is not needed.

## Usage

Run the standard single-pass baseline:

```bash
python run_experiment.py --standard-only
```

Run entropy-gated and energy-gated SEER:

```bash
python run_experiment.py --seer-only
```

Run all configurations:

```bash
python run_experiment.py
```

Restrict execution to one model:

```bash
python run_experiment.py --model qwen
python run_experiment.py --model deepseek
```

The experiment driver calibrates missing thresholds automatically and caches them in `results/calibration.json`.

## Analysis

Print aggregate execution, validity, step, and convergence statistics:

```bash
python analyze.py
```

Evaluate entropy and energy as passive detectors of execution incorrectness:

```bash
python analyze_detector.py
```

The detector analysis reports:

- AUROC for predicting execution incorrectness
- execution accuracy at different retained coverage levels
- comparisons between first-position, sequence-mean, and sequence-maximum uncertainty
- comparisons between entropy and energy

## Notebook

`SEER.ipynb` contains a ready-to-run version of the workflow for Google Colab or a local Jupyter environment with GPU support.

## Project structure

```text
SEER/
├── analyze.py
├── analyze_detector.py
├── bird_loader.py
├── config.py
├── executor.py
├── hf_model.py
├── prompt_builder.py
├── run_experiment.py
├── seer_runner.py
├── SEER.ipynb
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

Main components:

- `config.py` — model identifiers, sampling, decoding, calibration, gate, and execution settings
- `bird_loader.py` — BIRD-SQL question loading, deterministic sampling, schema serialization, and database-path discovery
- `prompt_builder.py` — system and user prompt construction
- `hf_model.py` — model loading, greedy generation, SQL extraction, entropy, energy, and aggregation
- `seer_runner.py` — threshold calibration, feedback construction, candidate selection, and the SEER refinement loop
- `executor.py` — read-only SQLite execution, timeout handling, and result-set comparison
- `run_experiment.py` — experiment orchestration and result serialization
- `analyze.py` — aggregate result summaries
- `analyze_detector.py` — AUROC and accuracy-coverage analysis

## Configuration

The main settings are defined in `config.py`.

```python
MAX_SAMPLES = 150
SHUFFLE_SEED = 42

K_MAX = 5
CALIB_SIGMA = 2.0
GATE_REDUCTION = "mean"

EXEC_GUIDED = True

MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.1
LOAD_IN_4BIT = True
```

Important options:

- `GATE_REDUCTION`
  - `"first"`: uncertainty at the first generated position
  - `"mean"`: mean uncertainty over the generated sequence
  - `"max"`: maximum uncertainty over the generated sequence
- `EXEC_GUIDED`
  - `True`: execute each candidate and use SQLite errors as repair feedback
  - `False`: disable execution guidance for ablation
- `LOAD_IN_4BIT`
  - `True`: use 4-bit NF4 quantization
  - `False`: load the model in bfloat16 when sufficient memory is available

The reported paper experiments use sequence-mean gating with execution guidance enabled.

## Threshold calibration

Thresholds are calibrated independently for each model and gate.

SEER runs 10 deterministic textbook SQL questions over a built-in synthetic two-table schema. For each model and uncertainty signal, it computes the mean `mu` and population standard deviation `sigma` of the selected sequence reduction, then sets:

```text
tau = mu + 2 sigma
```

Calibration records are cached in:

```text
results/calibration.json
```

A cached threshold is recalculated when its stored reduction does not match the current `GATE_REDUCTION`.

## Output files

```text
results/
├── calibration.json
├── qwen/
│   ├── detail_standard.json
│   ├── detail_seer_entropy.json
│   ├── detail_seer_energy.json
│   ├── summary_standard.json
│   ├── summary_seer_entropy.json
│   └── summary_seer_energy.json
└── deepseek/
    ├── detail_standard.json
    ├── detail_seer_entropy.json
    ├── detail_seer_energy.json
    ├── summary_standard.json
    ├── summary_seer_entropy.json
    └── summary_seer_energy.json
```

Each detail record may include:

- predicted SQL
- gold SQL
- execution correctness
- SQL validity
- SQLite error
- refinement steps
- convergence status
- selected gate
- gate trace
- entropy and energy at first, mean, and maximum reductions
- generated token count
- whether the final output differs from the first candidate

## Reproducibility

The default experiment is reproducible under a fixed environment because:

- the BIRD-SQL subset is selected by deterministic shuffling with seed 42;
- all configurations use the same sampled questions;
- decoding is greedy with sampling disabled;
- thresholds are calibrated separately and cached;
- each query is evaluated against the same SQLite database and gold query;
- execution uses a fixed timeout;
- model weights remain frozen.

Hardware, package versions, GPU kernels, quantization backends, and model-library updates may still introduce small implementation-level differences.

## Evaluation semantics

The primary metric is execution accuracy. A prediction is considered correct when its result set matches the gold query result as an order-insensitive multiset of rows.

The implementation also reports:

- valid SQL rate
- average refinement steps
- gate convergence rate
- AUROC for detecting execution-incorrect predictions
- execution accuracy over retained confidence coverage

Execution success is not equivalent to logical correctness. A query may run successfully while returning the wrong result.

## Limitations

- Experiments use a fixed sample of 150 BIRD-SQL development questions by default.
- Execution feedback is effective for syntactic and schema-related failures but cannot reliably identify executable yet logically incorrect SQL.
- Threshold calibration uses a small synthetic reference set and does not provide formal conformal guarantees.
- Candidate selection prioritizes executability and uncertainty, not semantic correctness.
- Results may vary with model revisions, library versions, quantization, and hardware.
- The test split is not used for the reported experiments.

## Citation

```bibtex
@misc{alexander2026seer,
  title  = {SEER: Sequence Entropy--Energy Gating for Selective Prediction and Text-to-SQL Refinement},
  author = {Leonardo Alexander and Wilson Handojo and Meiliana and Rilo Chandra Pradana},
  year   = {2026},
  note   = {GitHub repository},
  url    = {https://github.com/leonardo-alexander/SEER}
}
```

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` before opening a pull request.

Changes affecting calibration, gate semantics, candidate selection, execution scoring, or reported metrics should clearly describe their behavioral impact.

## License

See `LICENSE` for license terms.
