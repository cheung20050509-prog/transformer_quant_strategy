import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ABL_ROOT = Path(__file__).resolve().parent
ROOT = ABL_ROOT.parent
TRANSFORMER_ROOT = ROOT / 'transformer_quant_strategy'
OUTPUT_ROOT = ABL_ROOT / 'output_signal_only'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.append(str(TRANSFORMER_ROOT))
from trading_strategy import TradingStrategy
from backtest_engine import BacktestEngine


def zscore_by_date(df: pd.DataFrame, col: str = 'predicted') -> pd.DataFrame:
    out = df.copy()

    def _z(group: pd.DataFrame) -> pd.DataFrame:
        std = group[col].std()
        mean = group[col].mean()
        if std is None or std < 1e-12:
            group[col] = 0.0
        else:
            group[col] = (group[col] - mean) / std
        return group

    return out.groupby('date', group_keys=False).apply(_z)


def build_signal_only_scores(feature_data: pd.DataFrame) -> pd.DataFrame:
    pred = feature_data[['date', 'stock_code']].copy()
    if 'net_signal' in feature_data.columns:
        raw = feature_data['net_signal'].astype(float)
    elif {'bull_signal_strength', 'bear_signal_strength'}.issubset(feature_data.columns):
        raw = feature_data['bull_signal_strength'].astype(float) - feature_data['bear_signal_strength'].astype(float)
    else:
        momentum = feature_data['momentum_5'].astype(float) if 'momentum_5' in feature_data.columns else 0.0
        macd_hist = feature_data['macd_hist'].astype(float) if 'macd_hist' in feature_data.columns else 0.0
        raw = momentum + macd_hist
    pred['predicted'] = raw
    return pred


def metrics_on_range(eq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    data = eq[(eq['date'] >= start) & (eq['date'] <= end)].copy().sort_values('date')
    if data.empty:
        return {
            'n_days': 0,
            'total_return': np.nan,
            'annual_return': np.nan,
            'max_drawdown': np.nan,
            'sharpe_ratio': np.nan,
        }

    data['ret'] = data['total_equity'].pct_change().fillna(0.0)
    total_return = data['total_equity'].iloc[-1] / data['total_equity'].iloc[0] - 1
    annual_return = (1 + total_return) ** (250 / len(data)) - 1
    peak = data['total_equity'].cummax()
    max_drawdown = ((peak - data['total_equity']) / peak).max()
    annual_vol = data['ret'].std() * np.sqrt(250)
    sharpe = (data['ret'].mean() * 250 - 0.0275) / (annual_vol + 1e-10)

    return {
        'n_days': int(len(data)),
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe),
    }


def main() -> None:
    feature_path = TRANSFORMER_ROOT / 'output' / 'feature_data.csv'
    params_path = TRANSFORMER_ROOT / 'output' / 'optuna_best_params.json'
    main_metrics_path = TRANSFORMER_ROOT / 'output' / 'performance_metrics.csv'
    main_trade_log_path = TRANSFORMER_ROOT / 'output' / 'trade_log.csv'

    if not feature_path.exists():
        raise FileNotFoundError(f'Missing file: {feature_path}')
    if not params_path.exists():
        raise FileNotFoundError(f'Missing file: {params_path}')
    if not main_metrics_path.exists():
        raise FileNotFoundError(f'Missing file: {main_metrics_path}')
    if not main_trade_log_path.exists():
        raise FileNotFoundError(f'Missing file: {main_trade_log_path}')

    feature_data = pd.read_csv(feature_path)
    feature_data['date'] = pd.to_datetime(feature_data['date'])

    main_trade_log = pd.read_csv(main_trade_log_path)
    main_trade_log['date'] = pd.to_datetime(main_trade_log['date'])
    window_start = main_trade_log['date'].min()
    window_end = main_trade_log['date'].max()

    feature_window = feature_data[
        (feature_data['date'] >= window_start) & (feature_data['date'] <= window_end)
    ].copy()
    if feature_window.empty:
        raise RuntimeError('No feature data found in the main experiment date window.')

    signal_pred = build_signal_only_scores(feature_window)
    signal_pred = zscore_by_date(signal_pred, col='predicted')

    with open(params_path, 'r', encoding='utf-8') as f:
        strategy_params = json.load(f)['strategy']

    strategy = TradingStrategy(
        initial_capital=1000000,
        max_position_ratio=float(strategy_params['max_position_ratio']),
        stop_loss_ratio=float(strategy_params['stop_loss_ratio']),
        take_profit_ratio=float(strategy_params['take_profit_ratio']),
        use_kelly=True,
        buy_threshold=float(strategy_params['buy_threshold']),
        sell_threshold=float(strategy_params['sell_threshold']),
    )

    signals_sig = strategy.generate_signals(predictions=signal_pred, feature_data=feature_data)
    backtest = BacktestEngine(initial_capital=1000000)
    signal_results = backtest.run_backtest(signals=signals_sig, price_data=feature_data)
    eq_sig = signal_results['equity_curve'].copy()
    eq_sig['date'] = pd.to_datetime(eq_sig['date'])

    metrics_sig = metrics_on_range(eq_sig, window_start, window_end)

    main_metrics = pd.read_csv(main_metrics_path).iloc[0]
    metrics_it = {
        'n_days': int(metrics_sig['n_days']),
        'total_return': float(main_metrics['total_return']),
        'annual_return': float(main_metrics['annual_return']),
        'max_drawdown': float(main_metrics['max_drawdown']),
        'sharpe_ratio': float(main_metrics['sharpe_ratio']),
    }

    comparison = pd.DataFrame([
        {'model': 'iTransformer_main_official', 'start': window_start.date(), 'end': window_end.date(), **metrics_it},
        {'model': 'SignalOnly_NoModel_same_window', 'start': window_start.date(), 'end': window_end.date(), **metrics_sig},
    ])

    out_path = OUTPUT_ROOT / 'fair_main_window_comparison.csv'
    comparison.to_csv(out_path, index=False)

    eq_sig.to_csv(OUTPUT_ROOT / 'signal_only_equity_curve_main_window.csv', index=False)
    signals_sig.to_csv(OUTPUT_ROOT / 'signal_only_signals_main_window.csv', index=False)

    print('MAIN_WINDOW', window_start.date(), window_end.date())
    print(comparison.to_string(index=False))
    print('saved:', out_path)


if __name__ == '__main__':
    main()
