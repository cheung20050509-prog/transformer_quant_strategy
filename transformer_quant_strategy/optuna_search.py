# -*- coding: utf-8 -*-
"""
Optuna 两阶段超参数搜索
========================
阶段1: 固定策略参数(用已知最优值)，搜索9个模型超参数
阶段2: 固定阶段1最优模型参数，搜索5个策略超参数

改进:
- 统一数据库文件 optuna_study.db，study名区分阶段
- 日志追加模式，不覆盖历史
- 每个trial结果即时保存到CSV（防崩溃丢数据）
- n_startup_trials=5，合理的随机/引导比例
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import csv
import argparse
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import torch

# 导入项目模块
from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineering
from transformer_model import TransformerPredictor
from trading_strategy import TradingStrategy
from backtest_engine import BacktestEngine

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 统一数据库路径
DB_PATH = os.path.join(OUTPUT_DIR, 'optuna_study.db')
STORAGE = f'sqlite:///{DB_PATH}'

# 每个trial即时保存的CSV
TRIAL_LOG_CSV = os.path.join(OUTPUT_DIR, 'optuna_trial_log.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Trial 2 最优策略参数（阶段1固定使用）
BEST_STRATEGY_PARAMS = {
    'buy_threshold': 0.01184,
    'sell_threshold': -0.01016,
    'stop_loss_ratio': 0.03109,
    'take_profit_ratio': 0.14665,
    'max_position_ratio': 0.33254,
}

# === Trial 71 最优参数（Phase2, 100 trials, 2026-02-27）===
# 年化40.68%, 回撤5.43%, Sharpe 1.197, 胜率42.47%
TRIAL71_STRATEGY_PARAMS = {
    'buy_threshold': 0.01534,
    'sell_threshold': -0.00798,
    'stop_loss_ratio': 0.06238,
    'take_profit_ratio': 0.14166,
    'max_position_ratio': 0.37682,
}
TRIAL71_MODEL_PARAMS = {
    'd_model': 512,
    'd_ff': 2048,
    'n_heads': 8,
    'n_layers': 16,
    'seq_length': 60,
    'dropout': 0.21538,
    'learning_rate': 3.046e-04,
    'batch_size': 512,
    'epochs': 800,
}

# === NO_LEAK最优参数（Val-scored, 100stocks updated, Phase1=20t+Phase2=100t）===
# 验证集: 年化14.57%, 回撤1.96%, Sharpe 1.331, 胜率79.38%
# 测试集: 年化13.47%, 回撤2.99%, Sharpe 1.254, 胜率45.83%
NOLEAK_BEST_STRATEGY_PARAMS = {
    'buy_threshold': 0.02387,
    'sell_threshold': -0.00432,
    'stop_loss_ratio': 0.05876,
    'take_profit_ratio': 0.08208,
    'max_position_ratio': 0.34502,
}
NOLEAK_BEST_MODEL_PARAMS = {
    'd_model': 1536,
    'd_ff': 3072,
    'n_heads': 32,
    'n_layers': 14,
    'seq_length': 60,
    'dropout': 0.20656,
    'learning_rate': 2.450e-04,
    'batch_size': 64,
    'epochs': 440,
}

# Trial 2 最优模型参数（用于enqueue，作为搜索基准）
BEST_MODEL_PARAMS_SEED = {
    'd_model': 1536,
    'd_ff_ratio': 4,
    'n_heads': 8,
    'n_layers': 16,
    'seq_length': 30,
    'dropout': 0.22106,
    'learning_rate': 5.595e-05,
    'batch_size': 512,
    'epochs': 390,
}

# Phase1 最优模型参数（Optuna两阶段搜索 Trial 8, d_model=512, Sharpe=0.926）
# 用于 --skip-phase1 直接进入Phase2
PHASE1_BEST_MODEL_PARAMS = {
    'd_model': 512,
    'd_ff': 2048,  # d_ff_ratio=4
    'n_heads': 8,
    'n_layers': 16,
    'seq_length': 60,
    'dropout': 0.21538,
    'learning_rate': 3.046e-04,
    'batch_size': 512,
    'epochs': 800,
}

# 股票池（100只，与main.py一致）
ALL_STOCKS = [
    # ========== 消费板块（15只）==========
    '600519.SH', '000858.SZ', '000568.SZ', '600887.SH', '601888.SH',
    '000895.SZ', '603288.SH', '002304.SZ', '000661.SZ', '600809.SH',
    '603369.SH', '600132.SH', '002714.SZ', '600600.SH', '000876.SZ',
    # ========== 金融板块（15只）==========
    '601318.SH', '600036.SH', '000001.SZ', '601166.SH', '600030.SH',
    '601688.SH', '601398.SH', '601939.SH', '600016.SH', '601601.SH',
    '601628.SH', '600837.SH', '601211.SH', '000776.SZ', '601377.SH',
    # ========== 制造/科技板块（20只）==========
    '000333.SZ', '000651.SZ', '002415.SZ', '600031.SH', '600588.SH',
    '002230.SZ', '601012.SH', '002074.SZ', '002352.SZ', '601100.SH',
    '000725.SZ', '002371.SZ', '600690.SH', '000100.SZ',
    '600406.SH', '300014.SZ', '002594.SZ', '601766.SH', '600104.SH',
    # ========== 医药板块（15只）==========
    '600276.SH', '000538.SZ', '601607.SH', '000963.SZ',
    '600196.SH', '002007.SZ', '300122.SZ', '600085.SH', '002001.SZ',
    '000423.SZ', '600436.SH', '300015.SZ', '002603.SZ', '300347.SZ',
    # ========== 能源/材料（15只）==========
    '600900.SH', '600309.SH', '601857.SH', '600028.SH', '601088.SH',
    '600585.SH', '002460.SZ', '600346.SH', '601899.SH', '600547.SH',
    '002466.SZ', '600188.SH', '601985.SH', '601225.SH', '601898.SH',
    # ========== 地产/基建/交运（10只）==========
    '600048.SH', '001979.SZ', '601668.SH', '601390.SH', '600115.SH',
    '601111.SH', '600029.SH', '601006.SH', '600009.SH', '601288.SH',
    # ========== 通信/传媒（10只）==========
    '600050.SH', '600522.SH', '000063.SZ', '002475.SZ', '600183.SH',
    '002236.SZ', '300413.SZ', '002049.SZ', '600745.SH',
]


# ============================================================
# 数据准备（只做一次）
# ============================================================
def prepare_data(stock_list, start_date='20160101', end_date='20251231'):
    """准备数据，返回 feature_data 和 feature_cols"""
    print("=" * 60)
    print("[数据准备] 获取并预处理数据（所有trial共享）")
    print("=" * 60)

    data_module = DataAcquisition()
    print("\n[1] 获取股票数据...")
    all_stock_data = data_module.fetch_stock_data(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )

    print("\n[2] 数据清洗...")
    cleaned_data = data_module.clean_data(all_stock_data)

    print("\n[3] 特征工程...")
    fe_module = FeatureEngineering()
    feature_data = fe_module.compute_all_features(cleaned_data)
    feature_cols = fe_module.get_feature_columns()

    n_stocks = len(feature_data['stock_code'].unique())
    print(f"\n  数据准备完成: {n_stocks}只股票, {len(feature_data)}条, {len(feature_cols)}个特征")

    return feature_data, feature_cols


# ============================================================
# 评估函数
# ============================================================
def evaluate_trial(feature_data, feature_cols, model_params, strategy_params):
    """训练模型 + 在验证集上回测，返回绩效指标（防超参数过拟合：不用测试集选参）"""
    try:
        predictor = TransformerPredictor(
            seq_length=model_params['seq_length'],
            pred_length=5,
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            n_layers=model_params['n_layers'],
            d_ff=model_params['d_ff'],
            dropout=model_params['dropout'],
            epochs=model_params['epochs'],
            learning_rate=model_params['learning_rate'],
            batch_size=model_params['batch_size'],
            use_itransformer=True
        )

        model_results = predictor.train_and_predict(
            feature_data=feature_data,
            feature_cols=feature_cols,
            target_col='future_return_5d',
            train_ratio=0.70,
            val_ratio=0.15
        )

        # === 用验证集预测做回测（Optuna选参）===
        val_predictions = model_results.get('val_predictions')
        if val_predictions is None or len(val_predictions) == 0:
            print("    [警告] 无验证集预测，回退到测试集")
            val_predictions = model_results['predictions']

        strategy = TradingStrategy(
            initial_capital=1000000,
            max_position_ratio=strategy_params['max_position_ratio'],
            stop_loss_ratio=strategy_params['stop_loss_ratio'],
            take_profit_ratio=strategy_params['take_profit_ratio'],
            use_kelly=True,
            buy_threshold=strategy_params['buy_threshold'],
            sell_threshold=strategy_params['sell_threshold']
        )

        signals = strategy.generate_signals(
            predictions=val_predictions,
            feature_data=feature_data
        )

        backtest = BacktestEngine(initial_capital=1000000)
        backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
        metrics = backtest.calculate_metrics(backtest_results)

        metrics['direction_accuracy'] = model_results.get('val_direction_accuracy', 0)
        metrics['mse'] = model_results.get('val_mse', float('inf'))
        metrics['best_val_loss'] = model_results.get('best_val_loss', float('inf'))
        metrics['best_epoch'] = model_results.get('best_epoch', 0)
        metrics['final_train_loss'] = model_results.get('final_train_loss', float('inf'))

        return metrics

    except Exception as e:
        print(f"    [trial失败] {e}")
        import traceback
        traceback.print_exc()
        return None


def train_model_once(feature_data, feature_cols, model_params):
    """训练一次模型，返回预测结果（复用于Phase2策略搜索）"""
    print(f"\n{'='*60}")
    print(f"  [Phase2前置] 使用最优模型参数训练一次模型...")
    print(f"  d_model={model_params['d_model']}, d_ff={model_params['d_ff']}, "
          f"layers={model_params['n_layers']}, epochs={model_params['epochs']}")
    print(f"{'='*60}")

    predictor = TransformerPredictor(
        seq_length=model_params['seq_length'],
        pred_length=5,
        d_model=model_params['d_model'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        d_ff=model_params['d_ff'],
        dropout=model_params['dropout'],
        epochs=model_params['epochs'],
        learning_rate=model_params['learning_rate'],
        batch_size=model_params['batch_size'],
        use_itransformer=True
    )

    model_results = predictor.train_and_predict(
        feature_data=feature_data,
        feature_cols=feature_cols,
        target_col='future_return_5d',
        train_ratio=0.70,
        val_ratio=0.15
    )

    print(f"  模型训练完成! direction_accuracy={model_results.get('direction_accuracy', 0):.4f}, "
          f"mse={model_results.get('mse', float('inf')):.6f}")

    # 保存预测结果到磁盘（供 main.py 加载，避免重新训练导致不一致）
    import pickle
    cache_path = os.path.join(OUTPUT_DIR, 'cached_model_results.pkl')
    cache_data = {
        'predictions': model_results['predictions'],           # 测试集预测 DataFrame
        'val_predictions': model_results.get('val_predictions'),  # 验证集预测 DataFrame
        'mse': model_results.get('mse'),
        'mae': model_results.get('mae'),
        'direction_accuracy': model_results.get('direction_accuracy'),
        'val_mse': model_results.get('val_mse'),
        'val_direction_accuracy': model_results.get('val_direction_accuracy'),
        'stock_codes': model_results.get('stock_codes'),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"  预测结果已缓存: {cache_path}")

    return model_results


def normalize_predictions(predictions_df):
    """将预测值z-score标准化，消除分布偏移对策略阈值的影响
    
    问题: 模型预测均值=-0.0085, 仅24%为正值，而实际涨跌比≈45%
    buy_threshold=0.024 在raw预测空间是+1.9σ的极端事件，几乎不触发
    
    修复: 对每个日期截面做z-score标准化 → 阈值在相对空间工作
    标准化后 buy_threshold=0.5 表示"预测收益高于当日平均0.5个标准差的股票"
    """
    df = predictions_df.copy()
    # 按日期做截面标准化（每天的股票间相对排序）
    def zscore_group(group):
        mean = group['predicted'].mean()
        std = group['predicted'].std()
        if std > 1e-10:
            group['predicted'] = (group['predicted'] - mean) / std
        else:
            group['predicted'] = 0.0
        return group
    df = df.groupby('date', group_keys=False).apply(zscore_group)
    return df


def evaluate_strategy_only(feature_data, predictions, strategy_params, model_extras,
                           normalize=True):
    """仅评估策略参数（使用已有预测结果，不重新训练模型）"""
    try:
        # 预测值z-score标准化，让阈值在相对空间工作
        if normalize:
            pred_input = normalize_predictions(predictions)
        else:
            pred_input = predictions

        strategy = TradingStrategy(
            initial_capital=1000000,
            max_position_ratio=strategy_params['max_position_ratio'],
            stop_loss_ratio=strategy_params['stop_loss_ratio'],
            take_profit_ratio=strategy_params['take_profit_ratio'],
            use_kelly=True,
            buy_threshold=strategy_params['buy_threshold'],
            sell_threshold=strategy_params['sell_threshold']
        )

        signals = strategy.generate_signals(
            predictions=pred_input,
            feature_data=feature_data
        )

        backtest = BacktestEngine(initial_capital=1000000)
        backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
        metrics = backtest.calculate_metrics(backtest_results)

        metrics['direction_accuracy'] = model_extras.get('direction_accuracy', 0)
        metrics['mse'] = model_extras.get('mse', float('inf'))

        return metrics

    except Exception as e:
        print(f"    [trial失败] {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_strategy_cv(feature_data, predictions, strategy_params, model_extras,
                         n_splits=3):
    """分时段交叉验证评估策略（防止策略只适配单一市场环境）
    
    将验证期切成n_splits个子窗口，分别回测取平均Sharpe/年化/回撤
    这样optimizer无法找到只在某段行情有效的参数
    """
    try:
        pred_norm = normalize_predictions(predictions)
        dates = sorted(pred_norm['date'].unique())
        n_dates = len(dates)
        split_size = n_dates // n_splits
        
        sharpe_list = []
        annual_ret_list = []
        max_dd_list = []
        win_rate_list = []
        total_trades_list = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else n_dates
            window_dates = dates[start_idx:end_idx]
            
            window_preds = pred_norm[pred_norm['date'].isin(window_dates)]
            if len(window_preds) == 0:
                continue
            
            strategy = TradingStrategy(
                initial_capital=1000000,
                max_position_ratio=strategy_params['max_position_ratio'],
                stop_loss_ratio=strategy_params['stop_loss_ratio'],
                take_profit_ratio=strategy_params['take_profit_ratio'],
                use_kelly=True,
                buy_threshold=strategy_params['buy_threshold'],
                sell_threshold=strategy_params['sell_threshold']
            )
            
            signals = strategy.generate_signals(
                predictions=window_preds,
                feature_data=feature_data
            )
            
            backtest = BacktestEngine(initial_capital=1000000)
            backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
            metrics = backtest.calculate_metrics(backtest_results)
            
            sharpe_list.append(metrics.get('sharpe_ratio', -999))
            annual_ret_list.append(metrics.get('annual_return', -999))
            max_dd_list.append(metrics.get('max_drawdown', 999))
            win_rate_list.append(metrics.get('win_rate', 0))
            total_trades_list.append(metrics.get('total_trades', 0))
        
        if not sharpe_list:
            return None
        
        # 返回各窗口的平均指标
        avg_metrics = {
            'sharpe_ratio': np.mean(sharpe_list),
            'annual_return': np.mean(annual_ret_list),
            'max_drawdown': np.mean(max_dd_list),
            'win_rate': np.mean(win_rate_list),
            'total_trades': int(np.mean(total_trades_list)),
            'direction_accuracy': model_extras.get('direction_accuracy', 0),
            'mse': model_extras.get('mse', float('inf')),
            # 记录各窗口Sharpe方差（越小越稳定）
            'sharpe_std': np.std(sharpe_list),
        }
        return avg_metrics
    
    except Exception as e:
        print(f"    [CV评估失败] {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 即时保存每个trial结果到CSV（防崩溃丢数据）
# ============================================================
def save_trial_to_csv(phase, trial_num, params, metrics, elapsed):
    """每个trial完成后立即追加到CSV（使用固定列顺序避免对齐问题）"""
    # 固定列顺序，Phase1和Phase2共用同一套列名
    FIXED_COLUMNS = [
        'timestamp', 'phase', 'trial', 'elapsed_sec',
        # 模型参数（固定顺序）
        'd_model', 'd_ff', 'd_ff_ratio', 'n_heads', 'n_layers', 'seq_length',
        'dropout', 'learning_rate', 'batch_size', 'epochs',
        # 策略参数（固定顺序）
        'buy_threshold', 'sell_threshold', 'stop_loss_ratio',
        'take_profit_ratio', 'max_position_ratio',
        # 绩效指标（固定顺序）
        'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate',
        'total_trades', 'direction_accuracy', 'mse',
        # 训练损失信息
        'best_val_loss', 'best_epoch', 'final_train_loss',
    ]

    row = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase': phase,
        'trial': trial_num,
        'elapsed_sec': f"{elapsed:.0f}",
    }
    # 所有参数
    for k, v in params.items():
        if isinstance(v, float):
            row[k] = f"{v:.6f}"
        else:
            row[k] = v
    # 所有指标
    if metrics:
        for k in ['annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate',
                   'total_trades', 'direction_accuracy', 'mse',
                   'best_val_loss', 'best_epoch', 'final_train_loss']:
            v = metrics.get(k)
            if v is not None:
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
                else:
                    row[k] = v

    file_exists = os.path.exists(TRIAL_LOG_CSV)
    with open(TRIAL_LOG_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIXED_COLUMNS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"    [已保存] {TRIAL_LOG_CSV}")


# ============================================================
# Phase1专用: 纯模型评估（不做回测，不依赖策略参数）
# ============================================================
def evaluate_model_only(feature_data, feature_cols, model_params):
    """训练模型，返回预测能力指标（方向准确率、MSE、loss），不做回测"""
    try:
        predictor = TransformerPredictor(
            seq_length=model_params['seq_length'],
            pred_length=5,
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            n_layers=model_params['n_layers'],
            d_ff=model_params['d_ff'],
            dropout=model_params['dropout'],
            epochs=model_params['epochs'],
            learning_rate=model_params['learning_rate'],
            batch_size=model_params['batch_size'],
            use_itransformer=True
        )

        model_results = predictor.train_and_predict(
            feature_data=feature_data,
            feature_cols=feature_cols,
            target_col='future_return_5d',
            train_ratio=0.70,
            val_ratio=0.15
        )

        return {
            'direction_accuracy': model_results.get('val_direction_accuracy', 0),
            'mse': model_results.get('val_mse', float('inf')),
            'best_val_loss': model_results.get('best_val_loss', float('inf')),
            'best_epoch': model_results.get('best_epoch', 0),
            'final_train_loss': model_results.get('final_train_loss', float('inf')),
        }

    except Exception as e:
        print(f"    [trial失败] {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 阶段1: 模型超参数搜索
# ============================================================
def create_phase1_objective(feature_data, feature_cols):
    """阶段1目标函数：纯模型预测能力搜索（不依赖策略参数）
    
    优化目标: maximize(direction_accuracy - 10 * MSE)
    - 方向准确率: 模型预测涨跌方向的能力
    - MSE惩罚: 避免方向对但幅度误差大的模型
    """

    def objective(trial):
        # ===== 9个模型超参数 =====
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        d_ff_ratio = trial.suggest_categorical('d_ff_ratio', [2, 4])
        d_ff = d_ff_ratio * d_model
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
        n_layers = trial.suggest_int('n_layers', 2, 8)
        seq_length = trial.suggest_categorical('seq_length', [20, 30, 60])
        dropout = trial.suggest_float('dropout', 0.10, 0.50)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 50, 300, step=10)

        if d_model % n_heads != 0:
            raise optuna.TrialPruned()

        model_params = {
            'seq_length': seq_length, 'd_model': d_model,
            'n_heads': n_heads, 'n_layers': n_layers,
            'd_ff': d_ff, 'dropout': dropout,
            'learning_rate': learning_rate, 'batch_size': batch_size,
            'epochs': epochs,
        }

        trial_num = trial.number + 1
        print(f"\n{'='*60}")
        print(f"  [阶段1-模型搜索] Trial {trial_num}")
        print(f"{'='*60}")
        print(f"  模型: d={d_model}, d_ff={d_ff}, heads={n_heads}, layers={n_layers}, "
              f"seq={seq_length}")
        print(f"  训练: dropout={dropout:.3f}, lr={learning_rate:.6f}, "
              f"bs={batch_size}, epochs={epochs}")
        print(f"  目标: 最大化方向准确率 (无回测，纯模型评估)")

        t0 = time.time()
        metrics = evaluate_model_only(feature_data, feature_cols, model_params)
        elapsed = time.time() - t0

        if metrics is None:
            raise optuna.TrialPruned()

        dir_acc = metrics.get('direction_accuracy', 0)
        mse = metrics.get('mse', float('inf'))
        best_vloss = metrics.get('best_val_loss', float('inf'))
        best_ep = metrics.get('best_epoch', 0)
        final_tloss = metrics.get('final_train_loss', float('inf'))

        print(f"\n  [阶段1] Trial {trial_num} 结果 ({elapsed:.0f}s):")
        print(f"    方向准确率: {dir_acc:.2%}")
        print(f"    MSE:        {mse:.6f}")
        print(f"    训练Loss:   {final_tloss:.6f}")
        print(f"    验证Loss:   {best_vloss:.6f} (best@epoch {best_ep})")
        sys.stdout.flush()

        # 即时保存到CSV
        all_params = {**model_params, 'd_ff_ratio': d_ff_ratio}
        save_trial_to_csv('phase1', trial_num, all_params, metrics, elapsed)

        # 保存到trial属性
        trial.set_user_attr('direction_accuracy', dir_acc)
        trial.set_user_attr('mse', mse)
        trial.set_user_attr('best_val_loss', best_vloss)
        trial.set_user_attr('elapsed_time', elapsed)

        # 优化目标: maximize(direction_accuracy - 10*MSE)
        # Optuna minimize, 所以取负
        score = dir_acc - 10 * mse
        return -score

    return objective


# ============================================================
# 阶段2: 策略超参数搜索（使用预计算预测，不重新训练模型）
# ============================================================
def create_phase2_objective(feature_data, feature_cols, best_model_params,
                            cached_predictions=None, cached_model_extras=None):
    """阶段2目标函数：固定最优模型参数，搜索策略参数
    
    如果提供了cached_predictions，则不重新训练模型（秒级评估）。
    否则每个trial会重新训练模型（~40分钟/trial）。
    """

    def objective(trial):
        # ===== 5个策略超参数（z-score标准化空间）=====
        # 注意: 预测值已做z-score标准化，阈值含义:
        #   buy_threshold=0.5 → 预测收益高于当日均值0.5σ时买入
        #   sell_threshold=-0.3 → 预测收益低于当日均值0.3σ时卖出
        buy_threshold = trial.suggest_float('buy_threshold', 0.2, 1.5)
        sell_threshold = trial.suggest_float('sell_threshold', -1.5, -0.2)
        stop_loss_ratio = trial.suggest_float('stop_loss_ratio', 0.02, 0.08)
        take_profit_ratio = trial.suggest_float('take_profit_ratio', 0.04, 0.15)
        max_position_ratio = trial.suggest_float('max_position_ratio', 0.10, 0.40)

        strategy_params = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_ratio': stop_loss_ratio,
            'take_profit_ratio': take_profit_ratio,
            'max_position_ratio': max_position_ratio,
        }

        trial_num = trial.number + 1
        print(f"\n{'='*60}")
        print(f"  [阶段2-策略搜索] Trial {trial_num}")
        print(f"{'='*60}")
        print(f"  模型: 固定 (d={best_model_params['d_model']}, "
              f"layers={best_model_params['n_layers']}, "
              f"epochs={best_model_params['epochs']})")
        print(f"  策略: buy={buy_threshold:.4f}, sell={sell_threshold:.4f}, "
              f"sl={stop_loss_ratio:.3f}, tp={take_profit_ratio:.3f}, "
              f"pos={max_position_ratio:.2f}")

        t0 = time.time()

        if cached_predictions is not None:
            # 快速模式：复用预计算的预测结果 + 分时段交叉验证
            metrics = evaluate_strategy_cv(
                feature_data, cached_predictions, strategy_params, cached_model_extras,
                n_splits=3
            )
        else:
            # 慢速模式：重新训练模型
            metrics = evaluate_trial(feature_data, feature_cols, best_model_params, strategy_params)

        elapsed = time.time() - t0

        if metrics is None:
            raise optuna.TrialPruned()

        sharpe = metrics.get('sharpe_ratio', -999)
        annual_ret = metrics.get('annual_return', -999)
        max_dd = metrics.get('max_drawdown', 999)
        win_rate = metrics.get('win_rate', 0)

        sharpe_std = metrics.get('sharpe_std', 0)

        print(f"\n  [阶段2] Trial {trial_num} 结果 ({elapsed:.0f}s):")
        print(f"    年化收益率: {annual_ret:.2%} (3窗口CV平均)")
        print(f"    最大回撤:   {max_dd:.2%}")
        print(f"    夏普比率:   {sharpe:.4f} ± {sharpe_std:.4f}")
        print(f"    胜率:       {win_rate:.2%}")

        # 即时保存到CSV
        all_params = {**best_model_params, **strategy_params}
        save_trial_to_csv('phase2', trial_num, all_params, metrics, elapsed)

        trial.set_user_attr('annual_return', annual_ret)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('elapsed_time', elapsed)

        # 加入Sharpe稳定性惩罚，防止策略只在某个子窗口表现好
        score = sharpe * 0.5 + annual_ret * 0.3 - max_dd * 0.2 - sharpe_std * 0.3
        return -score

    return objective


# ============================================================
# 结果汇总
# ============================================================
def print_study_summary(study, phase_name):
    """打印study汇总"""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"\n  [{phase_name}] 无完成的trial")
        return None

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  {phase_name} 结果汇总")
    print(f"{'='*60}")
    print(f"  完成: {len(completed)} trials")
    print(f"  最佳: Trial {best.number+1} (综合得分={-best.value:.4f})")

    ar = best.user_attrs.get('annual_return')
    md = best.user_attrs.get('max_drawdown')
    sr = best.user_attrs.get('sharpe_ratio')
    wr = best.user_attrs.get('win_rate')
    if ar is not None:
        print(f"    年化收益: {ar:.2%}")
    if md is not None:
        print(f"    最大回撤: {md:.2%}")
    if sr is not None:
        print(f"    夏普比率: {sr:.4f}")
    if wr is not None:
        print(f"    胜率:     {wr:.2%}")

    print(f"\n  最佳参数:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    # Top-5
    ranked = sorted(completed, key=lambda t: t.value)[:5]
    print(f"\n  Top-5:")
    print(f"  {'Trial':>6} | {'得分':>8} | {'年化收益':>10} | {'最大回撤':>10} | {'夏普':>8}")
    print(f"  {'-'*55}")
    for t in ranked:
        t_ar = t.user_attrs.get('annual_return', 0)
        t_md = t.user_attrs.get('max_drawdown', 0)
        t_sr = t.user_attrs.get('sharpe_ratio', 0)
        print(f"  #{t.number+1:>5} | {-t.value:>8.4f} | {t_ar:>9.2%} | {t_md:>9.2%} | {t_sr:>7.4f}")

    return best


def save_final_results(phase1_best, phase2_best, output_dir, best_model_params=None,
                       test_metrics=None):
    """保存最终最优参数到JSON和TXT"""
    if phase2_best:
        best = phase2_best
        phase = 'phase2'
    elif phase1_best:
        best = phase1_best
        phase = 'phase1'
    else:
        return

    model_keys = ['d_model', 'd_ff_ratio', 'n_heads', 'n_layers', 'seq_length',
                  'dropout', 'learning_rate', 'batch_size', 'epochs']
    strategy_keys = ['buy_threshold', 'sell_threshold', 'stop_loss_ratio',
                     'take_profit_ratio', 'max_position_ratio']

    config = {
        'source': phase,
        'trial': best.number + 1,
        'score': -best.value,
        'model': {},
        'strategy': {},
        'performance_val': {
            'annual_return': best.user_attrs.get('annual_return'),
            'max_drawdown': best.user_attrs.get('max_drawdown'),
            'sharpe_ratio': best.user_attrs.get('sharpe_ratio'),
            'win_rate': best.user_attrs.get('win_rate'),
        },
        'performance_test': {
            'annual_return': test_metrics.get('annual_return') if test_metrics else None,
            'max_drawdown': test_metrics.get('max_drawdown') if test_metrics else None,
            'sharpe_ratio': test_metrics.get('sharpe_ratio') if test_metrics else None,
            'win_rate': test_metrics.get('win_rate') if test_metrics else None,
        },
    }

    for k in model_keys:
        if k in best.params:
            config['model'][k] = best.params[k]
    if 'd_ff_ratio' in config['model'] and 'd_model' in config['model']:
        config['model']['d_ff'] = config['model']['d_ff_ratio'] * config['model']['d_model']

    # Phase 2 best 没有模型参数在 params 中，用传入的 best_model_params 补充
    if not config['model'] and best_model_params:
        config['model'] = {k: v for k, v in best_model_params.items()}

    for k in strategy_keys:
        if k in best.params:
            config['strategy'][k] = best.params[k]

    # 如果阶段1没有策略参数，补充固定值
    if phase == 'phase1':
        config['strategy'] = BEST_STRATEGY_PARAMS.copy()

    # JSON
    json_path = os.path.join(output_dir, 'optuna_best_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  最优参数JSON: {json_path}")

    # TXT
    txt_path = os.path.join(output_dir, 'optuna_best_params.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Optuna 两阶段搜索 - 最终最优参数 (来自{phase})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Trial: {config['trial']}\n")
        f.write(f"综合得分: {config['score']:.4f}\n\n")

        f.write("【模型参数】\n")
        for k, v in config['model'].items():
            if isinstance(v, float):
                f.write(f"  {k:20s} = {v:.6f}\n")
            else:
                f.write(f"  {k:20s} = {v}\n")

        f.write("\n【策略参数】\n")
        for k, v in config['strategy'].items():
            if isinstance(v, float):
                f.write(f"  {k:20s} = {v:.5f}\n")
            else:
                f.write(f"  {k:20s} = {v}\n")

        f.write("\n【回测结果 - 验证集（Optuna选参依据）】\n")
        perf = config['performance_val']
        if perf.get('annual_return') is not None:
            f.write(f"  年化收益率         = {perf['annual_return']:.2%}\n")
        if perf.get('max_drawdown') is not None:
            f.write(f"  最大回撤           = {perf['max_drawdown']:.2%}\n")
        if perf.get('sharpe_ratio') is not None:
            f.write(f"  夏普比率           = {perf['sharpe_ratio']:.4f}\n")
        if perf.get('win_rate') is not None:
            f.write(f"  胜率               = {perf['win_rate']:.2%}\n")

        f.write("\n【回测结果 - 测试集（真正样本外，仅评估一次）】\n")
        perf_test = config['performance_test']
        if perf_test.get('annual_return') is not None:
            f.write(f"  年化收益率         = {perf_test['annual_return']:.2%}\n")
        if perf_test.get('max_drawdown') is not None:
            f.write(f"  最大回撤           = {perf_test['max_drawdown']:.2%}\n")
        if perf_test.get('sharpe_ratio') is not None:
            f.write(f"  夏普比率           = {perf_test['sharpe_ratio']:.4f}\n")
        if perf_test.get('win_rate') is not None:
            f.write(f"  胜率               = {perf_test['win_rate']:.2%}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("评分公式: score = sharpe*0.5 + annual_return*0.3 - max_drawdown*0.2\n")
        f.write("=" * 70 + "\n")

    print(f"  最优参数TXT: {txt_path}")

    return config


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Optuna两阶段超参数搜索')
    parser.add_argument('--phase1-trials', type=int, default=20,
                        help='阶段1(模型搜索)次数 (默认20)')
    parser.add_argument('--phase2-trials', type=int, default=16,
                        help='阶段2(策略搜索)次数 (默认16)')
    parser.add_argument('--stocks', type=int, default=100,
                        help='使用前N只股票 (默认100)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='跳过阶段1，直接用已有最优模型参数进入阶段2')
    args = parser.parse_args()

    print("=" * 60)
    print("  Optuna 两阶段超参数搜索")
    print("=" * 60)
    print(f"  阶段1(模型): {args.phase1_trials} trials")
    print(f"  阶段2(策略): {args.phase2_trials} trials")
    print(f"  股票数量:    {args.stocks}")
    print(f"  数据库:      {DB_PATH}")
    print(f"  Trial日志:   {TRIAL_LOG_CSV}")
    print(f"  设备:        {device}")

    stock_list = ALL_STOCKS[:args.stocks]

    # 准备数据
    feature_data, feature_cols = prepare_data(stock_list)

    # ========================================
    # 阶段1: 模型超参数搜索
    # ========================================
    phase1_best = None

    if not args.skip_phase1:
        print(f"\n{'='*60}")
        print(f"  ===== 阶段1: 模型超参数搜索 ({args.phase1_trials} trials) =====")
        print(f"{'='*60}")
        print(f"  固定策略参数: {BEST_STRATEGY_PARAMS}")

        sampler1 = TPESampler(seed=42, n_startup_trials=5)
        study1 = optuna.create_study(
            direction='minimize',
            sampler=sampler1,
            study_name='phase1_model',
            storage=STORAGE,
            load_if_exists=True,
        )

        completed1 = len([t for t in study1.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
        if completed1 > 0:
            print(f"  [恢复] 已有 {completed1} 个trial，继续...")
            print(f"  当前最佳得分: {-study1.best_value:.4f}")
        else:
            # 注入小模型基准参数（适配920样本）
            print(f"  [Enqueue] 注入小模型基准参数...")
            study1.enqueue_trial({
                'd_model': 128,
                'd_ff_ratio': 4,
                'n_heads': 8,
                'n_layers': 4,
                'seq_length': 30,
                'dropout': 0.30,
                'learning_rate': 1e-4,
                'batch_size': 64,
                'epochs': 150,
            })

        objective1 = create_phase1_objective(feature_data, feature_cols)
        study1.optimize(objective1, n_trials=args.phase1_trials, gc_after_trial=True)

        phase1_best = print_study_summary(study1, "阶段1-模型搜索")
    else:
        print("\n  [跳过阶段1] 使用已保存的最优模型参数...")
        # 尝试从DB加载，如果DB不存在则用hardcoded常量
        try:
            study1 = optuna.load_study(
                study_name='phase1_model',
                storage=STORAGE,
            )
            phase1_best = print_study_summary(study1, "阶段1-模型搜索(已有)")
        except Exception as e:
            print(f"  数据库加载失败: {e}")
            print(f"  使用 PHASE1_BEST_MODEL_PARAMS 常量...")
            phase1_best = None

    # 提取阶段1最优模型参数
    if phase1_best is not None:
        best_model_params = {
            'd_model': phase1_best.params['d_model'],
            'd_ff': phase1_best.params.get('d_ff_ratio', 4) * phase1_best.params['d_model'],
            'n_heads': phase1_best.params['n_heads'],
            'n_layers': phase1_best.params['n_layers'],
            'seq_length': phase1_best.params['seq_length'],
            'dropout': phase1_best.params['dropout'],
            'learning_rate': phase1_best.params['learning_rate'],
            'batch_size': phase1_best.params['batch_size'],
            'epochs': phase1_best.params['epochs'],
        }
    else:
        # 使用hardcoded Phase1最优参数
        best_model_params = PHASE1_BEST_MODEL_PARAMS.copy()
        print("\n  使用hardcoded模型参数:")
        for k, v in best_model_params.items():
            print(f"    {k}: {v}")

    print(f"\n  阶段1最优模型参数 → 阶段2:")
    for k, v in best_model_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    # ========================================
    # 阶段2: 策略超参数搜索（快速模式：训练一次模型，复用预测结果）
    # ========================================
    print(f"\n{'='*60}")
    print(f"  ===== 阶段2: 策略超参数搜索 ({args.phase2_trials} trials) =====")
    print(f"{'='*60}")
    print(f"  固定模型: d={best_model_params['d_model']}, "
          f"d_ff={best_model_params['d_ff']}, "
          f"layers={best_model_params['n_layers']}, "
          f"epochs={best_model_params['epochs']}")
    print(f"  模式: 训练一次模型 → 复用预测结果 → 秒级策略扫描")

    # 训练一次模型，获取预测结果
    model_results = train_model_once(feature_data, feature_cols, best_model_params)
    # Phase2 Optuna评分用验证集预测（防超参数过拟合），测试集留到最后
    cached_predictions = model_results.get('val_predictions', model_results['predictions'])
    cached_test_predictions = model_results['predictions']  # 测试集预测，最后报告用
    cached_model_extras = {
        'direction_accuracy': model_results.get('val_direction_accuracy', 0),
        'mse': model_results.get('val_mse', float('inf')),
    }
    cached_test_extras = {
        'direction_accuracy': model_results.get('direction_accuracy', 0),
        'mse': model_results.get('mse', float('inf')),
    }

    sampler2 = TPESampler(seed=123, n_startup_trials=5)
    study2 = optuna.create_study(
        direction='minimize',
        sampler=sampler2,
        study_name='phase2_strategy',
        storage=STORAGE,
        load_if_exists=True,
    )

    completed2 = len([t for t in study2.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    if completed2 > 0:
        print(f"  [恢复] 已有 {completed2} 个trial，继续...")
        print(f"  当前最佳得分: {-study2.best_value:.4f}")
    else:
        # 注入z-score空间的合理基准参数（不用旧的raw阈值，已不兼容）
        print(f"  [Enqueue] 注入z-score空间基准策略参数...")
        study2.enqueue_trial({
            'buy_threshold': 0.5,
            'sell_threshold': -0.3,
            'stop_loss_ratio': 0.05,
            'take_profit_ratio': 0.08,
            'max_position_ratio': 0.30,
        })

    objective2 = create_phase2_objective(
        feature_data, feature_cols, best_model_params,
        cached_predictions=cached_predictions,
        cached_model_extras=cached_model_extras
    )
    study2.optimize(objective2, n_trials=args.phase2_trials, gc_after_trial=True)

    phase2_best = print_study_summary(study2, "阶段2-策略搜索(验证集)")

    # ========================================
    # 最终测试集评估（仅用一次，防超参数过拟合）
    # ========================================
    print(f"\n{'='*60}")
    print(f"  ===== 最终测试集评估（held-out test set）=====")
    print(f"{'='*60}")
    print(f"  注意：Optuna选参基于验证集，以下是真正样本外的测试集结果")

    best_strategy_params = phase2_best.params if phase2_best else NOLEAK_BEST_STRATEGY_PARAMS
    test_metrics = evaluate_strategy_only(
        feature_data, cached_test_predictions, best_strategy_params, cached_test_extras
    )
    if test_metrics:
        print(f"\n  [测试集] 最终结果:")
        print(f"    年化收益率: {test_metrics.get('annual_return', 0):.2%}")
        print(f"    最大回撤:   {test_metrics.get('max_drawdown', 0):.2%}")
        print(f"    夏普比率:   {test_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"    胜率:       {test_metrics.get('win_rate', 0):.2%}")
        print(f"    方向准确率: {test_metrics.get('direction_accuracy', 0):.4f}")
    else:
        print(f"  [测试集] 评估失败")
        test_metrics = {}

    # ========================================
    # 保存最终结果
    # ========================================
    save_final_results(phase1_best, phase2_best, OUTPUT_DIR, best_model_params,
                       test_metrics=test_metrics)

    print(f"\n{'='*60}")
    print(f"  两阶段搜索全部完成！")
    if not args.skip_phase1:
        total1 = len([t for t in study1.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  阶段1完成: {total1} trials")
    else:
        print(f"  阶段1: 跳过（使用已有最优参数）")
    total2 = len([t for t in study2.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  阶段2完成: {total2} trials")
    print(f"  所有trial记录: {TRIAL_LOG_CSV}")
    print(f"  最优参数: {OUTPUT_DIR}/optuna_best_params.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
