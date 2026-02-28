# -*- coding: utf-8 -*-
"""
可视化模块
==========
功能：
1. 累计收益曲线
2. 预测效果对比图
3. 交易信号图
4. 回撤曲线
5. K线图
"""

import pandas as pd
import numpy as np
import os
import shutil

# 首先清除 matplotlib 缓存
_cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(_cache_dir):
    shutil.rmtree(_cache_dir, ignore_errors=True)

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Font settings (English only for paper figures)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-whitegrid')


class Visualization:
    """可视化类"""
    
    def __init__(self, output_dir: str = 'output', figsize: tuple = (12, 6)):
        """
        参数:
            output_dir: 输出目录
            figsize: 图形大小
        """
        self.output_dir = output_dir
        self.figsize = figsize
    
    def plot_equity_curves(self, results_dict: Dict[str, Dict], 
                          filename: str = 'equity_curves.png'):
        """
        绘制累计收益曲线对比图
        
        参数:
            results_dict: 策略名称到回测结果的映射
            filename: 保存文件名
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B9A4D']
        
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            equity_curve = results.get('equity_curve', pd.DataFrame())
            
            if len(equity_curve) == 0:
                continue
            
            # 计算累计收益率
            initial_equity = equity_curve['total_equity'].iloc[0]
            cumulative_return = (equity_curve['total_equity'] / initial_equity - 1) * 100
            
            ax.plot(equity_curve['date'], cumulative_return, 
                   label=strategy_name, linewidth=2, color=colors[i % len(colors)])
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Return Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 格式化日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")
    
    def plot_prediction_comparison(self, model_results: Dict,
                                   filename: str = 'prediction_comparison.png'):
        """
        绘制预测值与实际值对比图
        
        参数:
            model_results: 模型结果字典
            filename: 保存文件名
        """
        predictions_df = model_results.get('predictions', pd.DataFrame())
        
        if len(predictions_df) == 0:
            print("  无预测数据，跳过预测对比图")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1: 预测值vs实际值散点图
        ax1 = axes[0]
        ax1.scatter(predictions_df['actual'], predictions_df['predicted'], 
                   alpha=0.5, s=20, c='#2E86AB')
        
        # 添加对角线
        min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
        max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction')
        
        ax1.set_xlabel('Actual Return', fontsize=12)
        ax1.set_ylabel('Predicted Return', fontsize=12)
        ax1.set_title('Predicted vs Actual Returns (Scatter)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加评估指标文本
        mse = model_results.get('mse', 0)
        mae = model_results.get('mae', 0)
        direction_acc = model_results.get('direction_accuracy', 0)
        
        textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nDir. Accuracy: {direction_acc:.2%}'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 图2: 时序对比图（取部分数据）
        ax2 = axes[1]
        
        # 取一只股票的数据
        sample_stock = predictions_df['stock_code'].iloc[0]
        sample_data = predictions_df[predictions_df['stock_code'] == sample_stock].head(100)
        
        x = range(len(sample_data))
        ax2.plot(x, sample_data['actual'], label='Actual Return', linewidth=1.5, color='#2E86AB')
        ax2.plot(x, sample_data['predicted'], label='Predicted Return', linewidth=1.5, 
                color='#F18F01', linestyle='--')
        
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Return', fontsize=12)
        ax2.set_title(f'Predicted vs Actual Returns ({sample_stock})', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")
    
    def plot_trading_signals(self, signals: pd.DataFrame, 
                            price_data: pd.DataFrame,
                            stock_code: Optional[str] = None,
                            filename: str = 'trading_signals.png'):
        """
        绘制交易信号图
        
        参数:
            signals: 信号DataFrame
            price_data: 价格数据DataFrame
            stock_code: 指定股票代码，None则取第一只
            filename: 保存文件名
        """
        if stock_code is None:
            stock_code = signals['stock_code'].iloc[0]
        
        # 筛选股票数据
        sig_data = signals[signals['stock_code'] == stock_code].copy()
        price_df = price_data[price_data['stock_code'] == stock_code].copy()
        
        # 合并数据
        merged = sig_data.merge(price_df[['date', 'close']], on='date', how='left', 
                               suffixes=('', '_price'))
        if 'close_price' in merged.columns:
            merged['close'] = merged['close_price']
        
        merged = merged.sort_values('date').reset_index(drop=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # 图1: 价格走势和交易信号
        ax1 = axes[0]
        ax1.plot(merged['date'], merged['close'], linewidth=1.5, 
                color='#2E86AB', label='Close Price')
        
        # Mark buy points
        buy_points = merged[merged['signal'] == 1]
        ax1.scatter(buy_points['date'], buy_points['close'], 
                   marker='^', s=100, c='green', label='Buy Signal', zorder=5)
        
        # Mark sell points
        sell_points = merged[merged['signal'] == -1]
        ax1.scatter(sell_points['date'], sell_points['close'], 
                   marker='v', s=100, c='red', label='Sell Signal', zorder=5)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price (CNY)', fontsize=12)
        ax1.set_title(f'{stock_code} Trading Signals', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # 图2: 预测收益率
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in merged['predicted']]
        ax2.bar(merged['date'], merged['predicted'], color=colors, alpha=0.7, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Predicted Return', fontsize=12)
        ax2.set_title('Model Predicted Return', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")
    
    def plot_drawdown(self, backtest_results: Dict, 
                     filename: str = 'drawdown.png'):
        """
        绘制回撤曲线
        
        参数:
            backtest_results: 回测结果
            filename: 保存文件名
        """
        equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        
        if len(equity_curve) == 0:
            print("  无权益数据，跳过回撤图")
            return
        
        # 计算回撤
        equity = equity_curve['total_equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # 百分比
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                                gridspec_kw={'height_ratios': [2, 1]})
        
        # 图1: 权益曲线
        ax1 = axes[0]
        ax1.plot(equity_curve['date'], equity / 10000, linewidth=1.5, color='#2E86AB')
        ax1.fill_between(equity_curve['date'], equity / 10000, alpha=0.3, color='#2E86AB')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Equity (10k CNY)', fontsize=12)
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 回撤曲线
        ax2 = axes[1]
        ax2.fill_between(equity_curve['date'], 0, -drawdown, 
                        color='#C73E1D', alpha=0.7)
        ax2.plot(equity_curve['date'], -drawdown, color='#C73E1D', linewidth=0.5)
        
        # 标记最大回撤点
        max_dd_idx = np.argmax(drawdown)
        ax2.axhline(y=-drawdown[max_dd_idx], color='black', linestyle='--', 
                   linewidth=1, alpha=0.7)
        ax2.annotate(f'Max Drawdown: {drawdown[max_dd_idx]:.1f}%', 
                    xy=(equity_curve['date'].iloc[max_dd_idx], -drawdown[max_dd_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")
    
    def plot_attention_weights(self, model_results: Dict,
                              filename: str = 'attention_weights.png'):
        """
        可视化模型注意力权重（Transformer特有）
        
        这里绘制特征重要性替代图
        """
        predictions_df = model_results.get('predictions', pd.DataFrame())
        
        if len(predictions_df) == 0:
            print("  无模型数据，跳过注意力权重图")
            return
        
        # 由于Transformer的注意力权重提取较复杂，这里绘制预测分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 图1: 预测收益率分布
        ax1 = axes[0]
        ax1.hist(predictions_df['predicted'], bins=50, color='#2E86AB', 
                alpha=0.7, edgecolor='white')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Predicted Returns', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 预测vs实际的误差分布
        ax2 = axes[1]
        errors = predictions_df['predicted'] - predictions_df['actual']
        ax2.hist(errors, bins=50, color='#F18F01', alpha=0.7, edgecolor='white')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = errors.mean()
        std_error = errors.std()
        textstr = f'Mean: {mean_error:.6f}\nStd: {std_error:.6f}'
        ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")
    
    def plot_strategy_comparison_table(self, comparison_df: pd.DataFrame,
                                       filename: str = 'strategy_comparison_table.png'):
        """
        绘制策略对比表格图
        
        参数:
            comparison_df: 策略对比DataFrame
            filename: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(
            cellText=comparison_df.values,
            colLabels=comparison_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # 设置表头样式
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # 设置第一列样式
        for i in range(1, len(comparison_df) + 1):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(fontweight='bold')
        
        plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {self.output_dir}/{filename}")


# 测试代码
if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=250, freq='B')
    
    # 模拟回测结果
    equity1 = 100 * np.cumprod(1 + np.random.randn(250) * 0.01 + 0.0005)
    equity2 = 100 * np.cumprod(1 + np.random.randn(250) * 0.008 + 0.0003)
    equity3 = 100 * np.cumprod(1 + np.random.randn(250) * 0.012 + 0.0002)
    
    results1 = {
        'equity_curve': pd.DataFrame({'date': dates, 'total_equity': equity1})
    }
    results2 = {
        'equity_curve': pd.DataFrame({'date': dates, 'total_equity': equity2})
    }
    results3 = {
        'equity_curve': pd.DataFrame({'date': dates, 'total_equity': equity3})
    }
    
    # 测试可视化
    viz = Visualization(output_dir='output')
    
    viz.plot_equity_curves({
        'iTransformer Strategy': results1,
        'MA Crossover': results2,
        'Buy & Hold': results3
    })
    
    viz.plot_drawdown(results1)
    
    print("可视化测试完成！")
