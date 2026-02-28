# -*- coding: utf-8 -*-
"""
Optuna两阶段超参数搜索脚本
===========================
策略：
  阶段1: 固定策略参数（使用已知最优值），只搜模型超参数（9个）
  阶段2: 固定最优模型参数，只搜策略超参数（5个）

这样将14维联合搜索拆解为 9维+5维，大幅降低搜索难度，
避免TPE在高维噪声空间中迷失。
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
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

# 输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 已知最优策略参数（来自Trial 2, 42.07%年化）
# 阶段1搜索时使用这组固定值
# ============================================================
DEFAULT_STRATEGY_PARAMS = {
    'buy_threshold': 0.01184,
    'sell_threshold': -0.01016,
    'stop_loss_ratio': 0.03109,
    'take_profit_ratio': 0.14665,
    'max_position_ratio': 0.33254,
}


# ============================================================
# 数据准备（只做一次，所有trial共享）
# ============================================================
def prepare_data(stock_list, start_date='20180101', end_date='20251231'):
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
    
    print(f"\n  数据准备完成:")
    print(f"    股票数量: {len(feature_data['stock_code'].unique())}")
    print(f"    数据总量: {len(feature_data)} 条")
    print(f"    特征数量: {len(feature_cols)}")
    
    return feature_data, feature_cols


# ============================================================
# 单次trial评估函数
# ============================================================
def evaluate_trial(feature_data, feature_cols, model_params, strategy_params):
    """
    用给定超参数训练模型 + 回测，返回绩效指标
    """
    try:
        # 1. 训练模型
        predictor = TransformerPredictor(
            seq_length=model_params['seq_length'],
            pred_length=5,
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            n_layers=model_params['n_layers'],
            d_ff=model_params.get('d_ff', 4 * model_params['d_model']),
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
            train_ratio=0.8
        )
        
        # 2. 生成交易信号
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
            predictions=model_results['predictions'],
            feature_data=feature_data
        )
        
        # 3. 回测
        backtest = BacktestEngine(initial_capital=1000000)
        backtest_results = backtest.run_backtest(signals=signals, price_data=feature_data)
        metrics = backtest.calculate_metrics(backtest_results)
        
        # 附加模型指标
        metrics['direction_accuracy'] = model_results.get('direction_accuracy', 0)
        metrics['mse'] = model_results.get('mse', float('inf'))
        
        return metrics
        
    except Exception as e:
        print(f"    [trial失败] {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_score(metrics):
    """统一的评分函数: score = sharpe*0.5 + annual_ret*0.3 - max_dd*0.2"""
    sharpe = metrics.get('sharpe_ratio', -999)
    annual_ret = metrics.get('annual_return', -999)
    max_dd = metrics.get('max_drawdown', 999)
    return sharpe * 0.5 + annual_ret * 0.3 - max_dd * 0.2


def print_trial_result(trial_num, phase, metrics, elapsed):
    """打印trial结果"""
    sharpe = metrics.get('sharpe_ratio', -999)
    annual_ret = metrics.get('annual_return', -999)
    max_dd = metrics.get('max_drawdown', 999)
    win_rate = metrics.get('win_rate', 0)
    score = compute_score(metrics)
    
    print(f"\n  [{phase}] Trial {trial_num} 结果 ({elapsed:.0f}s):")
    print(f"    年化收益率: {annual_ret:.2%}")
    print(f"    最大回撤: {max_dd:.2%}")
    print(f"    夏普比率: {sharpe:.4f}")
    print(f"    胜率: {win_rate:.2%}")
    print(f"    综合得分: {score:.4f}")


# ============================================================
# 阶段1: 搜索模型超参数（固定策略参数）
# ============================================================
def create_phase1_objective(feature_data, feature_cols):
    """阶段1目标函数: 只搜模型超参数，策略参数固定为已知最优值"""
    
    def objective(trial):
        # ========== 模型超参数搜索空间（9个）==========
        d_model = trial.suggest_categorical('d_model', [512, 1024, 1536, 2048])
        n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
        n_layers = trial.suggest_int('n_layers', 4, 16)
        seq_length = trial.suggest_categorical('seq_length', [30, 60, 90, 120])
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        d_ff_ratio = trial.suggest_categorical('d_ff_ratio', [2, 4])
        epochs = trial.suggest_int('epochs', 200, 800, step=10)
        
        d_ff = d_ff_ratio * d_model
        
        # 确保 d_model 能被 n_heads 整除
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()
        
        model_params = {
            'seq_length': seq_length, 'd_model': d_model,
            'n_heads': n_heads, 'n_layers': n_layers, 'd_ff': d_ff,
            'dropout': dropout, 'learning_rate': learning_rate,
            'batch_size': batch_size, 'epochs': epochs,
        }
        
        # 策略参数固定（来自Trial 2最优值）
        strategy_params = DEFAULT_STRATEGY_PARAMS.copy()
        
        trial_num = trial.number + 1
        print(f"\n{'='*60}")
        print(f"  [阶段1-模型搜索] Trial {trial_num}")
        print(f"{'='*60}")
        print(f"  模型: d={d_model}, d_ff={d_ff}, heads={n_heads}, layers={n_layers}, "
              f"seq={seq_length}, dropout={dropout:.2f}, lr={learning_rate:.6f}, "
              f"bs={batch_size}, epochs={epochs}")
        print(f"  策略: 固定（buy={strategy_params['buy_threshold']:.4f}, "
              f"sell={strategy_params['sell_threshold']:.4f}）")
        
        t0 = time.time()
        metrics = evaluate_trial(feature_data, feature_cols, model_params, strategy_params)
        elapsed = time.time() - t0
        
        if metrics is None:
            raise optuna.TrialPruned()
        
        print_trial_result(trial_num, "阶段1", metrics, elapsed)
        
        # 记录指标
        trial.set_user_attr('annual_return', metrics.get('annual_return', -999))
        trial.set_user_attr('max_drawdown', metrics.get('max_drawdown', 999))
        trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
        trial.set_user_attr('sharpe_ratio', metrics.get('sharpe_ratio', -999))
        trial.set_user_attr('elapsed_time', elapsed)
        
        score = compute_score(metrics)
        return -score  # Optuna minimize → 取负
    
    return objective


# ============================================================
# 阶段2: 搜索策略超参数（固定模型参数）
# ============================================================
def create_phase2_objective(feature_data, feature_cols, best_model_params):
    """阶段2目标函数: 固定最优模型参数，只搜策略超参数"""
    
    def objective(trial):
        # ========== 策略超参数搜索空间（5个）==========
        buy_threshold = trial.suggest_float('buy_threshold', 0.003, 0.025)
        sell_threshold = trial.suggest_float('sell_threshold', -0.025, -0.002)
        stop_loss_ratio = trial.suggest_float('stop_loss_ratio', 0.02, 0.08)
        take_profit_ratio = trial.suggest_float('take_profit_ratio', 0.04, 0.20)
        max_position_ratio = trial.suggest_float('max_position_ratio', 0.10, 0.45)
        
        strategy_params = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_ratio': stop_loss_ratio,
            'take_profit_ratio': take_profit_ratio,
            'max_position_ratio': max_position_ratio,
        }
        
        # 模型参数固定
        model_params = best_model_params.copy()
        
        trial_num = trial.number + 1
        print(f"\n{'='*60}")
        print(f"  [阶段2-策略搜索] Trial {trial_num}")
        print(f"{'='*60}")
        print(f"  模型: 固定（d={model_params['d_model']}, layers={model_params['n_layers']}）")
        print(f"  策略: buy={buy_threshold:.4f}, sell={sell_threshold:.4f}, "
              f"sl={stop_loss_ratio:.3f}, tp={take_profit_ratio:.3f}, "
              f"pos={max_position_ratio:.2f}")
        
        t0 = time.time()
        metrics = evaluate_trial(feature_data, feature_cols, model_params, strategy_params)
        elapsed = time.time() - t0
        
        if metrics is None:
            raise optuna.TrialPruned()
        
        print_trial_result(trial_num, "阶段2", metrics, elapsed)
        
        # 记录指标
        trial.set_user_attr('annual_return', metrics.get('annual_return', -999))
        trial.set_user_attr('max_drawdown', metrics.get('max_drawdown', 999))
        trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
        trial.set_user_attr('sharpe_ratio', metrics.get('sharpe_ratio', -999))
        trial.set_user_attr('elapsed_time', elapsed)
        
        score = compute_score(metrics)
        return -score
    
    return objective


# ============================================================
# 结果汇总与保存
# ============================================================
def summarize_study(study, phase_name, output_dir='output'):
    """汇总一个study的结果"""
    print(f"\n{'='*60}")
    print(f"  {phase_name} 结果汇总")
    print(f"{'='*60}")
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  没有完成的trial")
        return None
    
    best = study.best_trial
    
    print(f"\n  最佳Trial: #{best.number + 1}")
    print(f"  综合得分: {-best.value:.4f}")
    print(f"  年化收益: {best.user_attrs.get('annual_return', 'N/A'):.2%}")
    print(f"  最大回撤: {best.user_attrs.get('max_drawdown', 'N/A'):.2%}")
    print(f"  夏普比率: {best.user_attrs.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"  胜率: {best.user_attrs.get('win_rate', 'N/A'):.2%}")
    
    print(f"\n  最佳超参数:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    
    # 保存所有trial结果
    trials_data = []
    for t in completed:
        row = {'trial': t.number + 1, 'score': -t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        trials_data.append(row)
    
    df = pd.DataFrame(trials_data)
    df = df.sort_values('score', ascending=False)
    
    prefix = 'phase1_model' if '阶段1' in phase_name else 'phase2_strategy'
    save_path = os.path.join(output_dir, f'optuna_{prefix}_trials.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n  结果已保存: {save_path}")
    
    # 打印Top-5
    print(f"\n  Top-5:")
    print(f"  {'Trial':>6} | {'Score':>8} | {'年化收益':>10} | {'最大回撤':>10} | {'夏普':>8}")
    print(f"  {'-'*55}")
    for _, row in df.head(5).iterrows():
        ar = row.get('annual_return', 0)
        md = row.get('max_drawdown', 0)
        sr = row.get('sharpe_ratio', 0)
        print(f"  #{int(row['trial']):>5} | {row['score']:>8.4f} | "
              f"{ar:>9.2%} | {md:>9.2%} | {sr:>7.4f}")
    
    return best


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Optuna两阶段超参数搜索')
    parser.add_argument('--phase1-trials', type=int, default=50,
                        help='阶段1（模型搜索）试验次数 (默认50)')
    parser.add_argument('--phase2-trials', type=int, default=40,
                        help='阶段2（策略搜索）试验次数 (默认40)')
    parser.add_argument('--stocks', type=int, default=100,
                        help='使用前N只股票 (默认100)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='跳过阶段1，直接用已知最优模型参数进入阶段2')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Optuna 两阶段超参数搜索")
    print("=" * 60)
    print(f"  阶段1（模型搜索）: {args.phase1_trials} trials")
    print(f"  阶段2（策略搜索）: {args.phase2_trials} trials")
    print(f"  股票数量: {args.stocks}")
    print(f"  设备: {device}")
    if args.skip_phase1:
        print(f"  ⚡ 跳过阶段1，直接进入阶段2")
    
    # ---------- 股票池（100只，与main.py一致）----------
    all_stocks = [
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
        '002230.SZ', '601012.SH', '300750.SZ', '002352.SZ', '601816.SH',
        '000725.SZ', '002371.SZ', '600690.SH', '000100.SZ', '601138.SH',
        '600406.SH', '300014.SZ', '002594.SZ', '601766.SH', '600104.SH',
        # ========== 医药板块（15只）==========
        '600276.SH', '000538.SZ', '300760.SZ', '603259.SH', '000963.SZ',
        '600196.SH', '002007.SZ', '300122.SZ', '600085.SH', '002001.SZ',
        '000423.SZ', '600436.SH', '300015.SZ', '002603.SZ', '300347.SZ',
        # ========== 能源/材料（15只）==========
        '600900.SH', '600309.SH', '601857.SH', '600028.SH', '601088.SH',
        '600585.SH', '002460.SZ', '600346.SH', '601899.SH', '600547.SH',
        '002466.SZ', '600188.SH', '601985.SH', '601225.SH', '600989.SH',
        # ========== 地产/基建/交运（10只）==========
        '600048.SH', '001979.SZ', '601668.SH', '601390.SH', '600115.SH',
        '601111.SH', '600029.SH', '601006.SH', '600009.SH', '601288.SH',
        # ========== 通信/传媒（10只）==========
        '600050.SH', '601728.SH', '000063.SZ', '002475.SZ', '600183.SH',
        '002236.SZ', '300413.SZ', '603501.SH', '002049.SZ', '600745.SH',
    ]
    
    stock_list = all_stocks[:args.stocks]
    
    # ---------- 准备数据（只做一次）----------
    feature_data, feature_cols = prepare_data(
        stock_list=stock_list,
        start_date='20180101',
        end_date='20251231'
    )
    
    # ============================================================
    # 阶段1: 搜索模型超参数
    # ============================================================
    if not args.skip_phase1:
        print(f"\n{'#'*60}")
        print(f"  阶段1: 搜索最优模型超参数（策略参数固定）")
        print(f"  固定策略: buy={DEFAULT_STRATEGY_PARAMS['buy_threshold']:.4f}, "
              f"sell={DEFAULT_STRATEGY_PARAMS['sell_threshold']:.4f}")
        print(f"{'#'*60}")
        
        db_path_p1 = os.path.join(OUTPUT_DIR, 'optuna_phase1.db')
        storage_p1 = f'sqlite:///{db_path_p1}'
        sampler_p1 = TPESampler(seed=42, n_startup_trials=10)
        
        study_p1 = optuna.create_study(
            direction='minimize',
            sampler=sampler_p1,
            study_name='phase1_model_search',
            storage=storage_p1,
            load_if_exists=True,
        )
        
        completed_p1 = len([t for t in study_p1.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE])
        if completed_p1 > 0:
            print(f"\n  [恢复] 已有 {completed_p1} 个trial，继续...")
            print(f"  当前最佳: {-study_p1.best_value:.4f}")
        
        objective_p1 = create_phase1_objective(feature_data, feature_cols)
        study_p1.optimize(objective_p1, n_trials=args.phase1_trials, gc_after_trial=True)
        
        best_p1 = summarize_study(study_p1, "阶段1-模型搜索", OUTPUT_DIR)
        
        # 提取最优模型参数
        best_model_params = {
            'd_model': best_p1.params['d_model'],
            'n_heads': best_p1.params['n_heads'],
            'n_layers': best_p1.params['n_layers'],
            'seq_length': best_p1.params['seq_length'],
            'd_ff': best_p1.params['d_ff_ratio'] * best_p1.params['d_model'],
            'dropout': best_p1.params['dropout'],
            'learning_rate': best_p1.params['learning_rate'],
            'batch_size': best_p1.params['batch_size'],
            'epochs': best_p1.params['epochs'],
        }
        
        # 保存阶段1最优模型参数
        p1_json = os.path.join(OUTPUT_DIR, 'phase1_best_model_params.json')
        with open(p1_json, 'w', encoding='utf-8') as f:
            json.dump(best_model_params, f, indent=2, ensure_ascii=False)
        print(f"\n  阶段1最优模型参数已保存: {p1_json}")
        
    else:
        # 跳过阶段1，从文件或默认值加载模型参数
        p1_json = os.path.join(OUTPUT_DIR, 'phase1_best_model_params.json')
        if os.path.exists(p1_json):
            with open(p1_json, 'r') as f:
                best_model_params = json.load(f)
            print(f"\n  从 {p1_json} 加载阶段1最优模型参数")
        else:
            # 使用Trial 2参数作为默认
            best_model_params = {
                'd_model': 1536, 'd_ff': 6144, 'n_heads': 8,
                'n_layers': 16, 'seq_length': 30, 'dropout': 0.22106,
                'learning_rate': 5.595e-05, 'batch_size': 512, 'epochs': 390,
            }
            print(f"\n  使用Trial 2默认模型参数")
        
        for k, v in best_model_params.items():
            print(f"    {k}: {v}")
    
    # ============================================================
    # 阶段2: 搜索策略超参数
    # ============================================================
    print(f"\n{'#'*60}")
    print(f"  阶段2: 搜索最优策略超参数（模型参数固定）")
    print(f"  固定模型: d={best_model_params['d_model']}, "
          f"layers={best_model_params['n_layers']}, "
          f"seq={best_model_params['seq_length']}")
    print(f"{'#'*60}")
    
    db_path_p2 = os.path.join(OUTPUT_DIR, 'optuna_phase2.db')
    storage_p2 = f'sqlite:///{db_path_p2}'
    sampler_p2 = TPESampler(seed=123, n_startup_trials=8)
    
    study_p2 = optuna.create_study(
        direction='minimize',
        sampler=sampler_p2,
        study_name='phase2_strategy_search',
        storage=storage_p2,
        load_if_exists=True,
    )
    
    completed_p2 = len([t for t in study_p2.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_p2 > 0:
        print(f"\n  [恢复] 已有 {completed_p2} 个trial，继续...")
        print(f"  当前最佳: {-study_p2.best_value:.4f}")
    
    objective_p2 = create_phase2_objective(feature_data, feature_cols, best_model_params)
    study_p2.optimize(objective_p2, n_trials=args.phase2_trials, gc_after_trial=True)
    
    best_p2 = summarize_study(study_p2, "阶段2-策略搜索", OUTPUT_DIR)
    
    # ============================================================
    # 最终汇总
    # ============================================================
    if best_p2:
        best_strategy_params = {k: best_p2.params[k] for k in [
            'buy_threshold', 'sell_threshold', 'stop_loss_ratio',
            'take_profit_ratio', 'max_position_ratio'
        ]}
        
        final_config = {
            'model': best_model_params,
            'strategy': best_strategy_params,
            'phase1_score': None,
            'phase2_score': -best_p2.value,
            'phase2_annual_return': best_p2.user_attrs.get('annual_return'),
            'phase2_max_drawdown': best_p2.user_attrs.get('max_drawdown'),
            'phase2_sharpe_ratio': best_p2.user_attrs.get('sharpe_ratio'),
        }
        
        final_json = os.path.join(OUTPUT_DIR, 'optuna_best_params.json')
        with open(final_json, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"  两阶段搜索完成！最终最优配置:")
        print(f"{'='*60}")
        print(f"\n  【模型参数】")
        for k, v in best_model_params.items():
            print(f"    {k}: {v}")
        print(f"\n  【策略参数】")
        for k, v in best_strategy_params.items():
            print(f"    {k}: {v}")
        print(f"\n  【最终性能】")
        print(f"    年化收益: {best_p2.user_attrs.get('annual_return', 0):.2%}")
        print(f"    最大回撤: {best_p2.user_attrs.get('max_drawdown', 0):.2%}")
        print(f"    夏普比率: {best_p2.user_attrs.get('sharpe_ratio', 0):.4f}")
        print(f"\n  最终配置已保存: {final_json}")
    
    print(f"\n  使用最佳参数运行完整实验:")
    print(f"    1. 查看: cat {OUTPUT_DIR}/optuna_best_params.json")
    print(f"    2. 更新 main.py 中的参数")
    print(f"    3. 运行: ./run.sh")


if __name__ == '__main__':
    main()
