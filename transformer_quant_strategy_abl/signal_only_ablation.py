# -*- coding: utf-8 -*-
"""
Ablation Study: Signal-Only Strategy (No iTransformer)
======================================================
- Uses engineered technical signals only (no model prediction)
- Reuses the same TradingStrategy and BacktestEngine for fair comparison
- Outputs metrics in the same style as transformer_quant_strategy/output
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ABL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ABL_ROOT.parent
BASELINE_ROOT = PROJECT_ROOT / "transformer_quant_strategy"
OUTPUT_DIR = ABL_ROOT / "output_signal_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASELINE_ROOT))

from trading_strategy import TradingStrategy  # noqa: E402
from backtest_engine import BacktestEngine  # noqa: E402


def zscore_by_date(df: pd.DataFrame, col: str = "predicted") -> pd.DataFrame:
    out = df.copy()

    def _z(group: pd.DataFrame) -> pd.DataFrame:
        std = group[col].std()
        mean = group[col].mean()
        if std is None or std < 1e-12:
            group[col] = 0.0
        else:
            group[col] = (group[col] - mean) / std
        return group

    return out.groupby("date", group_keys=False).apply(_z)


def build_signal_only_scores(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build pseudo-prediction scores from pure technical signals only.

    Priority:
    1) net_signal (if available)
    2) bull_signal_strength - bear_signal_strength
    3) simple fallback from momentum_5 and macd_hist
    """
    required = {"date", "stock_code"}
    if not required.issubset(feature_data.columns):
        raise ValueError("feature_data missing required columns: date, stock_code")

    pred = feature_data[["date", "stock_code"]].copy()

    if "net_signal" in feature_data.columns:
        raw = feature_data["net_signal"].astype(float)
    elif {"bull_signal_strength", "bear_signal_strength"}.issubset(feature_data.columns):
        raw = feature_data["bull_signal_strength"].astype(float) - feature_data["bear_signal_strength"].astype(float)
    else:
        momentum = feature_data["momentum_5"].astype(float) if "momentum_5" in feature_data.columns else 0.0
        macd_hist = feature_data["macd_hist"].astype(float) if "macd_hist" in feature_data.columns else 0.0
        raw = momentum + macd_hist

    pred["predicted"] = raw
    return pred


def main():
    feature_path = BASELINE_ROOT / "output" / "feature_data.csv"
    params_path = BASELINE_ROOT / "output" / "optuna_best_params.json"

    if not feature_path.exists():
        raise FileNotFoundError(f"feature_data not found: {feature_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"strategy params not found: {params_path}")

    print("=" * 70)
    print("Signal-Only Ablation (No iTransformer)")
    print("=" * 70)

    feature_data = pd.read_csv(feature_path)
    feature_data["date"] = pd.to_datetime(feature_data["date"])

    all_dates = sorted(feature_data["date"].unique())
    train_idx = int(len(all_dates) * 0.70)
    val_idx = int(len(all_dates) * 0.85)
    val_end_date = all_dates[val_idx - 1]

    test_data = feature_data[feature_data["date"] > val_end_date].copy()
    if test_data.empty:
        raise RuntimeError("No test data after chronological split.")

    pred_df = build_signal_only_scores(test_data)
    pred_df = zscore_by_date(pred_df, col="predicted")

    with open(params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    strategy_params = best_params.get("strategy", {})

    strategy = TradingStrategy(
        initial_capital=1000000,
        max_position_ratio=float(strategy_params.get("max_position_ratio", 0.3733)),
        stop_loss_ratio=float(strategy_params.get("stop_loss_ratio", 0.0270)),
        take_profit_ratio=float(strategy_params.get("take_profit_ratio", 0.14)),
        use_kelly=True,
        buy_threshold=float(strategy_params.get("buy_threshold", 1.258)),
        sell_threshold=float(strategy_params.get("sell_threshold", -0.895)),
    )

    signals = strategy.generate_signals(predictions=pred_df, feature_data=feature_data)

    backtest = BacktestEngine(initial_capital=1000000)
    backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
    perf = backtest.calculate_metrics(backtest_results)

    if "future_return_5d" in test_data.columns:
        eval_df = pred_df.merge(
            test_data[["date", "stock_code", "future_return_5d"]],
            on=["date", "stock_code"],
            how="left",
        )
        direction_accuracy = float(
            (np.sign(eval_df["predicted"]) == np.sign(eval_df["future_return_5d"]))
            .astype(float)
            .mean()
        )
    else:
        direction_accuracy = np.nan

    print("\n[Signal-Only Ablation Metrics]")
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    if not np.isnan(direction_accuracy):
        print(f"direction_accuracy (vs future_return_5d): {direction_accuracy:.6f}")

    pred_df.to_csv(OUTPUT_DIR / "signal_only_predictions.csv", index=False)
    signals.to_csv(OUTPUT_DIR / "signal_only_signals.csv", index=False)
    backtest_results["equity_curve"].to_csv(OUTPUT_DIR / "signal_only_equity_curve.csv", index=False)
    if not backtest_results["trade_log"].empty:
        backtest_results["trade_log"].to_csv(OUTPUT_DIR / "signal_only_trade_log.csv", index=False)

    metrics_df = pd.DataFrame([
        {
            "model": "SignalOnly_NoModel",
            "direction_accuracy": direction_accuracy,
            **perf,
        }
    ])
    metrics_df.to_csv(OUTPUT_DIR / "signal_only_performance_metrics.csv", index=False)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
