# DG: A Multi-Agent Latent World Model with Stackelberg-QRE Planning

> **Paper**: *Stackelberg World Model: Explicit Bounded-Rationality Game Response for Autonomous Driving Prediction*
> **Status**: Research code (under review)

---

## Overview

Standard RSSM-style world models absorb surrounding agents as implicit environment dynamics:
p(s_{t+1} | s_t, a_ego)   ← other agents implicitly absorbed

We upgrade other vehicles to **explicit bounded-rationality responders** via Quantal Response Equilibrium (QRE), and plan with the ego vehicle as a **Stackelberg leader**:
s_{t+1}  = f(s_t, a_ego, a_other)

a_other  ~ QRE(s_t, a_ego, λ_i)        ← explicit QRE response

λ_i      ~ q(λ | observation history)  ← per-agent rationality inference

a_ego*   = argmax_a E_λ E_{a_other} [V(s, a, a_other)]

### Four Core Modules

| ID | Module | File |
|----|--------|------|
| C1 | Decoupled RSSM | `src/models/rssm.py` |
| C2 | Per-agent λ inference network | `src/models/lambda_net.py` |
| C3 | Latent-space Logit-QRE | `src/models/qre_module.py` |
| C4 | Combined training loss | `src/models/losses.py` |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Elueen/DG.git
cd DG

# 2. Create conda environment
conda env create -f environment.yml
conda activate dg

# 3. Install package in editable mode
pip install -e .
```

---

## Data Preparation

### NGSIM (US-101 + I-80)
1. Download from [NGSIM official site](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
2. Place raw csv files under `data/raw/ngsim/`
3. Run preprocessing:
```bash
python scripts/preprocess_ngsim.py --input data/raw/ngsim --output data/processed/ngsim --hz 5
```

### highD
1. Download from [highD dataset](https://www.highd-dataset.com/)
2. Place under `data/raw/highD/`
3. Run preprocessing:
```bash
python scripts/preprocess_highd.py --input data/raw/highD --output data/processed/highd --hz 5
```

Both datasets are downsampled to **5 Hz**. Input: past 3 s of (x, y, vx, vy) for up to 6 surrounding agents. Output: future 5 s trajectory distribution.

---

## Training

```bash
# Full model
python scripts/train.py --config configs/experiment/full_model.yaml

# Ablation: fixed λ
python scripts/train.py --config configs/experiment/ablation_fixed_lambda.yaml

# Ablation: no QRE module
python scripts/train.py --config configs/experiment/ablation_no_qre.yaml
```

---

## Evaluation

```bash
# Evaluate on NGSIM test split
python scripts/evaluate.py --config configs/experiment/full_model.yaml --dataset ngsim

# Run all baselines
python scripts/run_baselines.py --dataset ngsim

# Full ablation table
python scripts/run_ablation.py --dataset ngsim
```

Metrics reported: **RMSE** (1/2/3/4/5 s), **NLL**, **FDE**, **MR**

---

## Baselines

| Method | Reference | File |
|--------|-----------|------|
| CS-LSTM | Alahi et al., 2016 | `src/models/baselines/cs_lstm.py` |
| PiP | Song et al., 2020 | `src/models/baselines/pip.py` |
| VRD | — | `src/models/baselines/vrd.py` |

---

## Project Structure
DG/

├── configs/          # Hydra yaml configs

├── data/             # Raw & processed data (not tracked by git)

├── src/

│   ├── data/         # Dataset & dataloader

│   ├── models/       # C1–C4 core modules + baselines

│   ├── training/     # Trainer, scheduler, callbacks

│   ├── evaluation/   # Metrics & comparison

│   └── utils/        # Logging, geometry, reproducibility

├── scripts/          # CLI entry points

├── notebooks/        # EDA & result visualization

├── tests/            # Unit tests

├── environment.yml

└── setup.py

---

## Citation

If you use this code, please cite:

```bibtex
@article{dg2025stackelberg,
  title   = {Stackelberg World Model: Explicit Bounded-Rationality Game Response for Autonomous Driving Prediction},
  author  = {Anonymous},
  journal = {under review},
  year    = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
