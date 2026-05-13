# PDE_Agent

An LLM-driven autonomous research agent for neural operator optimization on PDE benchmarks. The agent automatically runs the full scientific loop — hypothesis → experiment → evaluation → reflection — without human intervention.

---

## Overview

PDE_Agent targets **1D Burgers equation** long-horizon prediction using Fourier Neural Operators (FNO). Instead of manually tuning models, an LLM agent (Qwen3-14B) autonomously:

- Analyzes error diagnostics and physical bottlenecks
- Proposes physically-grounded improvement hypotheses
- Modifies training code and hyperparameters
- Runs experiments, evaluates results, and decides whether to adopt or roll back changes
- Logs the full research process in structured scientific records

---

## Project Structure

```
PDE_Agent/
├── repos/
│   └── agent/
│       ├── dataset.py          # Data pipeline: training set + val mixing, pushforward support
│       ├── model.py            # FNO1d architecture
│       ├── train.py            # DDP training with pushforward trick and candidate mechanism
│       ├── predict.py          # Inference, scoring (Seg1/2/3), error diagnostics
│       ├── tools.py            # Agent tool functions (train, infer, read logs)
│       └── orchestrator.py     # LLM agent loop: plan → train → eval → reflect
├── start_agent_session.sh      # Submit SLURM job (4× A40 GPU)
├── run_sandbox.sh              # Run scripts inside Apptainer container
├── evaluator.py                # Standalone evaluation script
├── probe_ckpt.py               # Checkpoint inspection utility
└── check_all.py                # Environment and data sanity checks
```

---

## Key Design Decisions

**Why val data is mixed into training**

The official training set (`1D_Burgers_Sols_Nu0.001.hdf5`) covers only `t = 0~1.95s` after downsampling, while inference requires prediction to `t = 9.96s` (coverage: ~20%). The first 80 samples of `task1_val.hdf5` (covering full `t = 0~9.96s`) are mixed into training so the model sees the long-horizon dynamics it will be evaluated on. The remaining 20 samples are reserved for local scoring.

**Pushforward trick**

Training uses multi-step rollout loss (default `k=3`) instead of single-step supervision. This forces the model to stay stable under autoregressive inference and suppresses error accumulation.

**Candidate mechanism**

`train.py` writes its best checkpoint to `best_model_candidate.pt` rather than directly overwriting `best_model.pt`. The orchestrator runs inference on the candidate and only promotes it to `best_model.pt` if the score improves by ≥ 0.2 points. Otherwise all files are rolled back.

**LLM context design**

Each iteration the LLM receives:
- Physical background (Burgers equation, shock formation, FNO limitations)
- Data coverage analysis (training covers only 20% of inference range)
- Full scoring formula with physical interpretation
- Error time-series diagnostics (extrapolation ratio, inflection step)
- Last 6 experiment records with outcomes

This gives the LLM enough context to reason physically rather than blindly tune hyperparameters.

---

## Scoring

Evaluation covers 190 predicted steps (steps 10–199), split into three segments:

| Segment | Steps | Physical time | Weight | Formula |
|---|---|---|---|---|
| Seg1 | 10–57 | 0.5–2.8s | 25% | `100 × exp(-20 × Rel-MSE)` |
| Seg2 | 57–105 | 2.8–5.2s | 25% | `100 × exp(-10 × Rel-MSE)` |
| Seg3 | 105–200 | 5.2–9.9s | 50% | `max(Lorentzian, Fréchet)` |

Where `Lorentzian = 100 / (1 + 10 × RMSE)` and `Fréchet = 50 × exp(-FD²)`.

Seg3 carries the highest weight and is the primary optimization target.

---

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- `h5py`, `numpy`, `transformers`
- 4× A40 GPU (48GB each) recommended
- SLURM + Apptainer for job submission

---

## Usage

**Submit a fresh run (clears history):**
```bash
bash start_agent_session.sh --reset --watch
```

**Continue from previous checkpoint:**
```bash
bash start_agent_session.sh --watch
```

**Run a script manually inside the container:**
```bash
bash run_sandbox.sh repos/agent/predict.py --val_only
```

**GPU allocation:**
- LLM inference: GPU 0 only (Qwen3-14B, ~28GB)
- DDP training: all 4 GPUs
- The two phases are strictly serial — LLM process exits and releases memory before training starts

---

## Data

The following data files are required but not included in this repository:

| File | Shape | Description |
|---|---|---|
| `1D_Burgers_Sols_Nu0.001.hdf5` | (10000, 201, 1024) | Training data, PDEBench official |
| `task1_val.hdf5` | (100, 200, 256) | Local validation set (downsampled) |
| `task1_test.hdf5` | (1000, 10, 256) | Test set initial conditions only |

Data source: [PDEBench](https://github.com/pdebench/PDEBench)

---

## Agent Output

After each run, the following files are produced under `repos/submission/`:

| File | Description |
|---|---|
| `task1_pred.hdf5` | Predictions, shape (1000, 200, 256) |
| `task1_time.csv` | Training and inference wall-clock time |
| `task1_logs.log` | Full structured research log (JSONL format) |

The research log records every iteration: hypothesis, experiment config, training metrics, evaluation scores, error diagnostics, and the LLM's reflection. Failed experiments and rollback decisions are preserved.

---

## Team

- guandi WANG
- zhouya SHEN

---

## License
