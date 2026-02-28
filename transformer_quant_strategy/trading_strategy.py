# -*- coding: utf-8 -*-
"""
交易策略模块
============
功能：
1. 基于Transformer预测结果生成交易信号
2. 使用凯利公式进行仓位管理
3. 风险控制（止损/止盈）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TradingStrategy:
    """
    量化交易策略类
    
    策略逻辑:
    1. 信号生成: 基于Transformer预测的未来收益率
       - 预测收益率 > 阈值 → 买入信号
       - 预测收益率 < -阈值 → 卖出信号
    
    2. 仓位管理: 凯利公式
       f* = (p × b - q) / b
       其中: p=胜率, q=1-p, b=盈亏比
    
    3. 风险控制:
       - 止损线: 亏损达到5%强制平仓
       - 止盈线: 盈利达到10%可选择获利了结
       - 单只股票最大仓位限制
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 max_position_ratio: float = 0.2,
                 stop_loss_ratio: float = 0.05,
                 take_profit_ratio: float = 0.10,
                 use_kelly: bool = True,
                 buy_threshold: float = 0.015,
                 sell_threshold: float = -0.01):
        """
        参数:
            initial_capital: 初始资金
            max_position_ratio: 单只股票最大仓位比例
            stop_loss_ratio: 止损比例
            take_profit_ratio: 止盈比例
            use_kelly: 是否使用凯利公式
            buy_threshold: 买入阈值（预测收益率）
            sell_threshold: 卖出阈值（预测收益率）
        """
        self.initial_capital = initial_capital
        self.max_position_ratio = max_position_ratio
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.use_kelly = use_kelly
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # 历史胜率和盈亏比（用于凯利公式）
        # 使用更保守的估计
        self.win_rate = 0.52
        self.profit_loss_ratio = 1.3
    
    def generate_signals(self,
                        predictions: pd.DataFrame,
                        feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            predictions: 预测结果DataFrame，包含 date, stock_code, predicted 列
            feature_data: 特征数据DataFrame
        
        返回:
            包含交易信号的DataFrame
        """
        print("  信号生成规则:")
        print(f"    - 买入阈值: 预测收益率 > {self.buy_threshold:.2%}")
        print(f"    - 卖出阈值: 预测收益率 < {self.sell_threshold:.2%}")
        print(f"    - 止损线: {self.stop_loss_ratio:.1%}")
        print(f"    - 止盈线: {self.take_profit_ratio:.1%}")
        print(f"    - 凯利公式: {'启用' if self.use_kelly else '禁用'}")
        
        # 合并预测结果和价格数据（包括七大王牌指标）
        merge_cols = ['date', 'stock_code', 'close', 'open', 'high', 'low', 
                     'rsi_12', 'macd_cross', 'volatility_20',
                     # 七大王牌指标
                     'macd_strong_buy', 'macd_strong_sell', 'macd_hist_expanding',
                     'ma_bullish_align', 'ma_bearish_align', 'price_above_ma60',
                     'kdj_golden_cross', 'kdj_death_cross', 'kdj_oversold', 'kdj_overbought',
                     'dmi_bullish', 'dmi_strong_trend',
                     'bull_signal_strength', 'bear_signal_strength', 'net_signal',
                     'vol_price_up', 'vol_price_down', 'boll_pct']
        
        # 只选择存在的列
        available_cols = [c for c in merge_cols if c in feature_data.columns]
        
        signals = predictions.merge(
            feature_data[available_cols],
            on=['date', 'stock_code'],
            how='left'
        )
        
        # 生成原始信号
        signals['raw_signal'] = 0
        signals.loc[signals['predicted'] > self.buy_threshold, 'raw_signal'] = 1   # 买入
        signals.loc[signals['predicted'] < self.sell_threshold, 'raw_signal'] = -1  # 卖出
        
        # 结合技术指标过滤信号
        signals = self._filter_signals(signals)
        
        # 计算仓位
        signals = self._calculate_position(signals)
        
        # 统计信号
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        hold_signals = (signals['signal'] == 0).sum()
        
        print(f"\n  信号统计:")
        print(f"    - 买入信号: {buy_signals} 次")
        print(f"    - 卖出信号: {sell_signals} 次")
        print(f"    - 持仓不变: {hold_signals} 次")
        
        return signals
    
    def _filter_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        使用七大王牌指标过滤和增强信号
        
        过滤规则:
        1. RSI超买(>75)时不买入，超卖(<25)时不卖出
        2. MACD金叉增强买入信号，死叉增强卖出
        3. 均线多头排列增强买入，空头排列增强卖出
        4. KDJ金叉/死叉信号增强
        5. DMI趋势确认
        6. 综合信号强度考量
        """
        signals = signals.copy()
        signals['signal'] = signals['raw_signal']
        signals['signal_strength'] = 1.0
        
        # === 1. RSI过滤 ===
        if 'rsi_12' in signals.columns:
            # 超买时不买入
            signals.loc[(signals['raw_signal'] == 1) & (signals['rsi_12'] > 75), 'signal'] = 0
            # 超卖时不卖出（反而可能是买入机会）
            signals.loc[(signals['raw_signal'] == -1) & (signals['rsi_12'] < 25), 'signal'] = 0
            # RSI在低位时增强买入信号
            signals.loc[(signals['raw_signal'] == 1) & (signals['rsi_12'] < 40), 'signal_strength'] *= 1.3
            # RSI在高位时增强卖出信号
            signals.loc[(signals['raw_signal'] == -1) & (signals['rsi_12'] > 60), 'signal_strength'] *= 1.3
        
        # === 2. MACD七大王牌信号增强 ===
        if 'macd_strong_buy' in signals.columns:
            # 零轴上金叉是强烈的买入信号
            signals.loc[(signals['raw_signal'] == 1) & (signals['macd_strong_buy'] == 1), 'signal_strength'] *= 1.5
        
        if 'macd_strong_sell' in signals.columns:
            # 零轴下死叉是强烈的卖出信号
            signals.loc[(signals['raw_signal'] == -1) & (signals['macd_strong_sell'] == 1), 'signal_strength'] *= 1.5
        
        if 'macd_hist_expanding' in signals.columns:
            # MACD红柱放大时增强买入信号
            signals.loc[(signals['raw_signal'] == 1) & (signals['macd_hist_expanding'] == 1), 'signal_strength'] *= 1.2
        
        if 'macd_cross' in signals.columns:
            # MACD金叉增强买入
            signals.loc[(signals['raw_signal'] == 1) & (signals['macd_cross'] == 1), 'signal_strength'] *= 1.3
            # MACD死叉增强卖出
            signals.loc[(signals['raw_signal'] == -1) & (signals['macd_cross'] == -1), 'signal_strength'] *= 1.3
        
        # === 3. 均线排列信号 ===
        if 'ma_bullish_align' in signals.columns:
            # 多头排列时买入信号增强
            signals.loc[(signals['raw_signal'] == 1) & (signals['ma_bullish_align'] == 1), 'signal_strength'] *= 1.4
        
        if 'ma_bearish_align' in signals.columns:
            # 空头排列时卖出信号增强
            signals.loc[(signals['raw_signal'] == -1) & (signals['ma_bearish_align'] == 1), 'signal_strength'] *= 1.4
        
        if 'price_above_ma60' in signals.columns:
            # 价格在60日均线上方时更倾向买入
            signals.loc[(signals['raw_signal'] == 1) & (signals['price_above_ma60'] == 1), 'signal_strength'] *= 1.2
            # 价格在60日均线下方时更倾向卖出
            signals.loc[(signals['raw_signal'] == -1) & (signals['price_above_ma60'] == 0), 'signal_strength'] *= 1.2
        
        # === 4. KDJ信号增强 ===
        if 'kdj_golden_cross' in signals.columns:
            # KDJ金叉增强买入
            signals.loc[(signals['raw_signal'] == 1) & (signals['kdj_golden_cross'] == 1), 'signal_strength'] *= 1.3
        
        if 'kdj_death_cross' in signals.columns:
            # KDJ死叉增强卖出
            signals.loc[(signals['raw_signal'] == -1) & (signals['kdj_death_cross'] == 1), 'signal_strength'] *= 1.3
        
        if 'kdj_oversold' in signals.columns:
            # KDJ超卖时买入信号增强
            signals.loc[(signals['raw_signal'] == 1) & (signals['kdj_oversold'] == 1), 'signal_strength'] *= 1.2
        
        if 'kdj_overbought' in signals.columns:
            # KDJ超买时卖出信号增强
            signals.loc[(signals['raw_signal'] == -1) & (signals['kdj_overbought'] == 1), 'signal_strength'] *= 1.2
        
        # === 5. DMI趋势确认 ===
        if 'dmi_bullish' in signals.columns and 'dmi_strong_trend' in signals.columns:
            # DMI确认趋势时增强信号
            bull_trend = (signals['dmi_bullish'] == 1) & (signals['dmi_strong_trend'] == 1)
            bear_trend = (signals['dmi_bullish'] == 0) & (signals['dmi_strong_trend'] == 1)
            
            signals.loc[(signals['raw_signal'] == 1) & bull_trend, 'signal_strength'] *= 1.3
            signals.loc[(signals['raw_signal'] == -1) & bear_trend, 'signal_strength'] *= 1.3
        
        # === 6. 综合信号强度考量 ===
        if 'net_signal' in signals.columns:
            # 综合多头信号强时增强买入
            signals.loc[(signals['raw_signal'] == 1) & (signals['net_signal'] > 3), 'signal_strength'] *= 1.3
            # 综合空头信号强时增强卖出
            signals.loc[(signals['raw_signal'] == -1) & (signals['net_signal'] < -3), 'signal_strength'] *= 1.3
            
            # 信号矛盾时减弱（预测买入但综合空头信号强）
            signals.loc[(signals['raw_signal'] == 1) & (signals['net_signal'] < -2), 'signal'] = 0
            signals.loc[(signals['raw_signal'] == -1) & (signals['net_signal'] > 2), 'signal'] = 0
        
        # === 7. 成交量确认 ===
        if 'vol_price_up' in signals.columns:
            # 放量上涨增强买入信号
            signals.loc[(signals['raw_signal'] == 1) & (signals['vol_price_up'] == 1), 'signal_strength'] *= 1.2
        
        if 'vol_price_down' in signals.columns:
            # 放量下跌增强卖出信号
            signals.loc[(signals['raw_signal'] == -1) & (signals['vol_price_down'] == 1), 'signal_strength'] *= 1.2
        
        # === 8. 布林带位置 ===
        if 'boll_pct' in signals.columns:
            # 价格在布林带下轨附近时买入信号增强
            signals.loc[(signals['raw_signal'] == 1) & (signals['boll_pct'] < 0.2), 'signal_strength'] *= 1.2
            # 价格在布林带上轨附近时卖出信号增强
            signals.loc[(signals['raw_signal'] == -1) & (signals['boll_pct'] > 0.8), 'signal_strength'] *= 1.2
        
        # === 9. 波动率过滤 ===
        if 'volatility_20' in signals.columns:
            high_vol = signals['volatility_20'] > signals['volatility_20'].quantile(0.85)
            # 高波动时适当减少仓位强度
            signals.loc[high_vol, 'signal_strength'] *= 0.8
        
        return signals
    
    def _calculate_position(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        计算仓位
        
        使用凯利公式:
        f* = (p × b - q) / b
        
        实际应用中采用"半凯利"策略以降低风险
        """
        signals = signals.copy()
        
        if self.use_kelly:
            # 凯利公式计算最优仓位
            p = self.win_rate
            q = 1 - p
            b = self.profit_loss_ratio
            
            kelly_fraction = (p * b - q) / b
            
            # 使用半凯利
            kelly_fraction = kelly_fraction * 0.5
            
            # 限制最大仓位
            kelly_fraction = min(kelly_fraction, self.max_position_ratio)
            kelly_fraction = max(kelly_fraction, 0)
            
            print(f"\n  凯利公式计算:")
            print(f"    - 历史胜率: {p:.2%}")
            print(f"    - 盈亏比: {b:.2f}")
            print(f"    - 理论最优仓位: {(p * b - q) / b:.2%}")
            print(f"    - 实际使用仓位(半凯利): {kelly_fraction:.2%}")
            
            # 根据预测强度调整仓位
            signals['position_size'] = 0.0
            
            # 买入时的仓位
            buy_mask = signals['signal'] == 1
            signals.loc[buy_mask, 'position_size'] = (
                kelly_fraction * 
                signals.loc[buy_mask, 'signal_strength'] * 
                (1 + signals.loc[buy_mask, 'predicted'] * 10)  # 预测越强，仓位越大
            )
            
            # 限制最大仓位
            signals['position_size'] = signals['position_size'].clip(0, self.max_position_ratio)
            
        else:
            # 固定仓位
            signals['position_size'] = 0.0
            signals.loc[signals['signal'] == 1, 'position_size'] = self.max_position_ratio
        
        return signals
    
    def update_kelly_params(self, trade_log: pd.DataFrame):
        """
        根据历史交易更新凯利公式参数
        """
        if trade_log is None or len(trade_log) == 0:
            return
        
        # 计算胜率
        winning_trades = (trade_log['profit'] > 0).sum()
        total_trades = len(trade_log)
        
        if total_trades > 10:
            self.win_rate = winning_trades / total_trades
            
            # 计算盈亏比
            avg_win = trade_log[trade_log['profit'] > 0]['profit'].mean()
            avg_loss = abs(trade_log[trade_log['profit'] < 0]['profit'].mean())
            
            if avg_loss > 0:
                self.profit_loss_ratio = avg_win / avg_loss
            
            print(f"  凯利参数更新:")
            print(f"    - 新胜率: {self.win_rate:.2%}")
            print(f"    - 新盈亏比: {self.profit_loss_ratio:.2f}")


class RiskManager:
    """
    风险管理器
    
    功能:
    1. 止损检查
    2. 止盈检查
    3. 最大回撤监控
    4. 仓位风险评估
    """
    
    def __init__(self,
                 stop_loss_ratio: float = 0.05,
                 take_profit_ratio: float = 0.10,
                 max_drawdown_limit: float = 0.15):
        """
        参数:
            stop_loss_ratio: 止损比例
            take_profit_ratio: 止盈比例
            max_drawdown_limit: 最大回撤限制
        """
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.max_drawdown_limit = max_drawdown_limit
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """
        检查是否触发止损
        
        返回: True表示需要止损
        """
        loss_ratio = (entry_price - current_price) / entry_price
        return loss_ratio >= self.stop_loss_ratio
    
    def check_take_profit(self, entry_price: float, current_price: float) -> bool:
        """
        检查是否触发止盈
        
        返回: True表示可以止盈
        """
        profit_ratio = (current_price - entry_price) / entry_price
        return profit_ratio >= self.take_profit_ratio
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> tuple:
        """
        计算回撤
        
        返回: (当前回撤, 最大回撤)
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        
        current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return current_drawdown, max_drawdown
    
    def should_reduce_position(self, current_drawdown: float) -> bool:
        """
        判断是否应该减仓
        
        当回撤接近限制时，建议减仓
        """
        return current_drawdown > self.max_drawdown_limit * 0.7


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100)
    
    predictions = pd.DataFrame({
        'date': dates,
        'stock_code': ['600519.SH'] * 100,
        'predicted': np.random.randn(100) * 0.02,
        'actual': np.random.randn(100) * 0.02
    })
    
    feature_data = pd.DataFrame({
        'date': dates,
        'stock_code': ['600519.SH'] * 100,
        'close': 100 + np.cumsum(np.random.randn(100)),
        'open': 100 + np.cumsum(np.random.randn(100)),
        'high': 105 + np.cumsum(np.random.randn(100)),
        'low': 95 + np.cumsum(np.random.randn(100)),
        'rsi_12': np.random.uniform(30, 70, 100),
        'macd_cross': np.random.choice([-1, 0, 1], 100),
        'volatility_20': np.random.uniform(0.1, 0.3, 100)
    })
    
    # 测试策略
    strategy = TradingStrategy(
        initial_capital=1000000,
        use_kelly=True
    )
    
    signals = strategy.generate_signals(predictions, feature_data)
    
    print("\n信号示例:")
    print(signals[['date', 'stock_code', 'predicted', 'signal', 'position_size']].head(20))
