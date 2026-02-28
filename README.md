# Commitment Submission Package

This directory is the final submission package for the course project and paper.

## Folder structure

- `paper_latex/`
  - Final paper source (`main.tex`, `references.bib`) and compiled PDF.

- `transformer_quant_strategy/`
  - Main iTransformer quantitative trading pipeline.
  - Main entry scripts:
    - `run.sh` (full experiment)
    - `run_optuna.sh` (two-phase Optuna hyperparameter search)
  - Key outputs in `output/`:
    - `optuna_best_params.json` / `optuna_best_params.txt` (final best hyperparameters)
    - `performance_metrics.csv` (main experiment final metrics)
    - `strategy_comparison.csv` (benchmark comparison)
    - `trade_log.csv` (trade records)

- `transformer_quant_strategy_abl/`
  - Ablation entry script:
    - `signal_only_ablation.py`
  - Kept ablation outputs in `output_signal_only/`:
    - `fair_main_window_comparison.csv`
    - `signal_only_equity_curve_main_window.csv`
    - `signal_only_signals_main_window.csv`

## Quick start

From `transformer_quant_strategy/`:

- Run full experiment:
  - `bash run.sh`

- Run Optuna search:
  - `bash run_optuna.sh`



