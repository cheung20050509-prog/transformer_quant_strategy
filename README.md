# A-Share Quantitative Trading Strategy (iTransformer)

This repository contains the final submission package for the course project:
**Design and Analysis of A-Share Quantitative Trading Strategy Based on iTransformer Model**.

It includes:

- Full paper source and compiled PDF
- Main iTransformer trading system
- Ablation experiment (`w/o iTransformer`) and outputs

## Repository Structure

- `paper_latex/`
  - `main.tex`, `references.bib`, and compiled `main.pdf`
  - Final paper source and build artifacts

- `transformer_quant_strategy/`
  - Main training + backtest pipeline
  - Key scripts:
    - `run.sh`: run full experiment pipeline
    - `run_optuna.sh`: run two-phase Optuna search
    - `main.py`: main entry for end-to-end experiment
    - `optuna_search.py`: two-phase hyperparameter optimization
  - Core outputs in `output/`:
    - `optuna_best_params.json`, `optuna_best_params.txt`
    - `performance_metrics.csv`
    - `strategy_comparison.csv`
    - `trade_log.csv`

- `transformer_quant_strategy_abl/`
  - `signal_only_ablation.py`
  - `output_signal_only/`:
    - `fair_main_window_comparison.csv`
    - `signal_only_equity_curve_main_window.csv`
    - `signal_only_signals_main_window.csv`

## Environment Setup

Install dependencies from repository root:

```bash
pip install -r requirements.txt
```

Python version recommendation:

- Python 3.10+

## Minimal Reproduction (Main Experiment, 3 Commands)

```bash
pip install -r requirements.txt
cd transformer_quant_strategy
bash run.sh
```

This will generate the main experiment outputs in `transformer_quant_strategy/output/`,
including `performance_metrics.csv`, `strategy_comparison.csv`, and `trade_log.csv`.

## Minimal Reproduction (Ablation, 2 Commands)

```bash
cd transformer_quant_strategy_abl
python signal_only_ablation.py
```

This will generate ablation outputs in `transformer_quant_strategy_abl/output_signal_only/`,
including `fair_main_window_comparison.csv`, `signal_only_equity_curve_main_window.csv`,
and `signal_only_signals_main_window.csv`.

## How to Run (Detailed)

### 1) Main Experiment

```bash
cd transformer_quant_strategy
bash run.sh
```

### 2) Optuna Hyperparameter Search

```bash
cd transformer_quant_strategy
bash run_optuna.sh
```

### 3) Ablation (Signal-only, No iTransformer)

```bash
cd transformer_quant_strategy_abl
python signal_only_ablation.py
```

## Notes

- `requirements.txt` is now located at the repository root (same level as this README).
- Main experiment output and ablation output are both already included for reproducibility.
- The paper includes the project GitHub repository information on the first page footer.



