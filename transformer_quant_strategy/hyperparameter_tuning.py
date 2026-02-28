# -*- coding: utf-8 -*-
"""
超参数优化模块
=============
功能：
1. 使用Optuna进行Transformer模型超参数优化
2. 使用Optuna进行交易策略参数优化
3. 支持多目标优化（收益率 + 夏普比率 + 最大回撤）
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional
import warnings
import logging

# 设置Optuna日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """超参数优化类"""
    
    def __init__(self, feature_data: pd.DataFrame, feature_cols: list,
                 train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        初始化
        
        参数:
            feature_data: 带特征的数据
            feature_cols: 特征列名列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.feature_data = feature_data
        self.feature_cols = feature_cols
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # 划分数据集
        self._split_data()
        
        self.best_model_params = None
        self.best_strategy_params = None
        self.study_history = []
    
    def _split_data(self):
        """划分训练/验证/测试集"""
        # 按时间排序
        self.feature_data = self.feature_data.sort_values(['stock_code', 'date'])
        
        # 获取所有唯一日期
        all_dates = self.feature_data['date'].unique()
        n_dates = len(all_dates)
        
        train_end = int(n_dates * self.train_ratio)
        val_end = int(n_dates * (self.train_ratio + self.val_ratio))
        
        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:val_end]
        test_dates = all_dates[val_end:]
        
        self.train_data = self.feature_data[self.feature_data['date'].isin(train_dates)]
        self.val_data = self.feature_data[self.feature_data['date'].isin(val_dates)]
        self.test_data = self.feature_data[self.feature_data['date'].isin(test_dates)]
        
        print(f"  数据划分完成:")
        print(f"    训练集: {len(self.train_data)} 条 ({len(train_dates)} 天)")
        print(f"    验证集: {len(self.val_data)} 条 ({len(val_dates)} 天)")
        print(f"    测试集: {len(self.test_data)} 条 ({len(test_dates)} 天)")
    
    def optimize_model_params(self, n_trials: int = 50, timeout: int = 600) -> Dict:
        """
        优化Transformer模型超参数
        
        参数:
            n_trials: 优化试验次数
            timeout: 超时时间（秒）
        
        返回:
            Dict: 最佳超参数
        """
        print(f"\n{'='*50}")
        print("开始Transformer模型超参数优化")
        print(f"  试验次数: {n_trials}, 超时: {timeout}秒")
        print(f"{'='*50}")
        
        # 导入Transformer模型
        from transformer_model import TransformerPredictor
        
        def objective(trial):
            """目标函数"""
            # 采样超参数
            params = {
                'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
                'n_layers': trial.suggest_int('n_layers', 2, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'seq_length': trial.suggest_categorical('seq_length', [20, 30, 40]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            }
            
            # 确保 d_model 能被 n_heads 整除
            if params['d_model'] % params['n_heads'] != 0:
                return float('inf')
            
            try:
                # 创建模型
                model = TransformerPredictor(
                    input_dim=len(self.feature_cols),
                    d_model=params['d_model'],
                    n_heads=params['n_heads'],
                    n_layers=params['n_layers'],
                    dropout=params['dropout'],
                    seq_length=params['seq_length'],
                    learning_rate=params['learning_rate'],
                    epochs=30  # 快速训练用于评估
                )
                
                # 准备序列数据
                X_train, y_train = self._prepare_sequences(
                    self.train_data, 
                    self.feature_cols, 
                    'future_return_5d',
                    params['seq_length']
                )
                
                X_val, y_val = self._prepare_sequences(
                    self.val_data,
                    self.feature_cols,
                    'future_return_5d',
                    params['seq_length']
                )
                
                if len(X_train) < params['batch_size']:
                    return float('inf')
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 在验证集上评估
                predictions = model.predict(X_val)
                
                # 计算多个指标
                # 1. 方向准确率
                direction_acc = np.mean(np.sign(predictions) == np.sign(y_val))
                
                # 2. MSE
                mse = np.mean((predictions - y_val) ** 2)
                
                # 3. IC (信息系数)
                ic = np.corrcoef(predictions, y_val)[0, 1]
                if np.isnan(ic):
                    ic = 0
                
                # 综合得分（最大化方向准确率和IC，最小化MSE）
                score = direction_acc * 0.4 + ic * 0.4 - mse * 0.2
                
                # Optuna默认最小化，所以返回负值
                return -score
                
            except Exception as e:
                print(f"    试验失败: {e}")
                return float('inf')
        
        # 创建study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )
        
        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 获取最佳参数
        self.best_model_params = study.best_params
        
        print(f"\n最佳模型超参数:")
        for key, value in self.best_model_params.items():
            print(f"  {key}: {value}")
        print(f"最佳得分: {-study.best_value:.4f}")
        
        return self.best_model_params
    
    def optimize_strategy_params(self, predictions: np.ndarray, 
                                 actual_returns: np.ndarray,
                                 n_trials: int = 30) -> Dict:
        """
        优化交易策略参数
        
        参数:
            predictions: 模型预测值
            actual_returns: 实际收益率
            n_trials: 优化试验次数
        
        返回:
            Dict: 最佳策略参数
        """
        print(f"\n{'='*50}")
        print("开始交易策略参数优化")
        print(f"{'='*50}")
        
        def objective(trial):
            """目标函数"""
            # 采样策略参数
            params = {
                'buy_threshold': trial.suggest_float('buy_threshold', 0.005, 0.03),
                'sell_threshold': trial.suggest_float('sell_threshold', -0.03, -0.005),
                'stop_loss': trial.suggest_float('stop_loss', 0.03, 0.10),
                'take_profit': trial.suggest_float('take_profit', 0.08, 0.20),
                'max_position': trial.suggest_float('max_position', 0.2, 0.5),
                'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 80),
                'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 35),
            }
            
            # 模拟简单回测
            score = self._simulate_strategy(predictions, actual_returns, params)
            
            # 最大化得分
            return -score
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_strategy_params = study.best_params
        
        print(f"\n最佳策略参数:")
        for key, value in self.best_strategy_params.items():
            print(f"  {key}: {value}")
        
        return self.best_strategy_params
    
    def _prepare_sequences(self, df: pd.DataFrame, feature_cols: list,
                          target_col: str, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """准备时序数据"""
        X_list = []
        y_list = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].sort_values('date')
            
            if len(stock_df) < seq_length + 1:
                continue
            
            features = stock_df[feature_cols].values
            targets = stock_df[target_col].values
            
            for i in range(seq_length, len(stock_df)):
                X_list.append(features[i-seq_length:i])
                y_list.append(targets[i])
        
        return np.array(X_list), np.array(y_list)
    
    def _simulate_strategy(self, predictions: np.ndarray, 
                          actual_returns: np.ndarray,
                          params: Dict) -> float:
        """
        模拟策略并计算综合得分
        """
        capital = 1.0
        position = 0.0
        trade_count = 0
        returns = []
        
        for i, (pred, actual_ret) in enumerate(zip(predictions, actual_returns)):
            # 生成信号
            if pred > params['buy_threshold'] and position == 0:
                # 买入
                position = params['max_position']
                trade_count += 1
            elif pred < params['sell_threshold'] and position > 0:
                # 卖出
                position = 0
                trade_count += 1
            
            # 计算收益
            daily_return = position * actual_ret
            capital *= (1 + daily_return)
            returns.append(daily_return)
        
        # 计算指标
        returns = np.array(returns)
        total_return = capital - 1
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(250)
        else:
            sharpe = 0
        
        # 最大回撤
        cummax = np.maximum.accumulate(np.cumprod(1 + returns))
        max_dd = np.max(1 - np.cumprod(1 + returns) / cummax)
        
        # 综合得分
        score = total_return * 0.4 + sharpe * 0.1 - max_dd * 0.3 + (trade_count > 10) * 0.2
        
        return score
    
    def get_optimized_config(self) -> Dict:
        """获取优化后的完整配置"""
        config = {
            'model': self.best_model_params or {},
            'strategy': self.best_strategy_params or {},
        }
        return config


def quick_optimize(feature_data: pd.DataFrame, feature_cols: list,
                   model_trials: int = 30, strategy_trials: int = 20) -> Dict:
    """
    快速超参数优化
    
    参数:
        feature_data: 特征数据
        feature_cols: 特征列
        model_trials: 模型优化试验次数
        strategy_trials: 策略优化试验次数
    
    返回:
        Dict: 优化后的配置
    """
    tuner = HyperparameterTuner(feature_data, feature_cols)
    
    # 优化模型参数
    model_params = tuner.optimize_model_params(n_trials=model_trials)
    
    return {
        'model': model_params,
        'strategy': tuner.best_strategy_params
    }


# 测试代码
if __name__ == '__main__':
    print("超参数优化模块测试")
    print("请从main.py运行完整优化流程")
