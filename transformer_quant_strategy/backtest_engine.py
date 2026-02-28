# -*- coding: utf-8 -*-
"""
回测引擎模块
============
功能：
1. 策略回测
2. 绩效指标计算
3. 对比策略实现（买入持有、传统均线）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    回测引擎
    
    功能:
    1. 执行策略回测
    2. 生成交易记录
    3. 计算绩效指标
    4. 实现对比策略
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.001,
                 stamp_tax: float = 0.001):
        """
        参数:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage: 滑点
            stamp_tax: 印花税（卖出时收取）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.stamp_tax = stamp_tax
    
    def run_backtest(self,
                    signals: pd.DataFrame,
                    price_data: pd.DataFrame) -> Dict:
        """
        运行回测
        
        参数:
            signals: 交易信号DataFrame
            price_data: 价格数据DataFrame
        
        返回:
            回测结果字典
        """
        # 初始化
        cash = self.initial_capital
        positions = {}  # {stock_code: {'shares': int, 'cost': float}}
        equity_curve = []
        trade_log = []
        daily_returns = []
        
        # 合并信号和价格数据
        data = signals.merge(
            price_data[['date', 'stock_code', 'close', 'open']],
            on=['date', 'stock_code'],
            how='left',
            suffixes=('', '_price')
        )
        
        # 确保使用正确的close列
        if 'close_price' in data.columns:
            data['trade_price'] = data['close_price']
        else:
            data['trade_price'] = data['close']
        
        # 按日期排序
        data = data.sort_values('date').reset_index(drop=True)
        
        dates = data['date'].unique()
        prev_equity = self.initial_capital
        
        for date in dates:
            day_data = data[data['date'] == date]
            
            for _, row in day_data.iterrows():
                stock_code = row['stock_code']
                signal = row['signal']
                position_size = row.get('position_size', 0.2)
                price = row['trade_price']
                
                if pd.isna(price) or price <= 0:
                    continue
                
                # 买入信号
                if signal == 1 and stock_code not in positions:
                    # 计算可买入金额
                    available_cash = cash * position_size
                    
                    # 考虑滑点和佣金
                    actual_price = price * (1 + self.slippage)
                    commission = available_cash * self.commission_rate
                    
                    # 计算可买入股数（100股整数倍）
                    shares = int((available_cash - commission) / actual_price / 100) * 100
                    
                    if shares >= 100:
                        cost = shares * actual_price + commission
                        cash -= cost
                        
                        positions[stock_code] = {
                            'shares': shares,
                            'cost': actual_price,
                            'entry_date': date
                        }
                        
                        trade_log.append({
                            'date': date,
                            'stock_code': stock_code,
                            'action': 'BUY',
                            'shares': shares,
                            'price': actual_price,
                            'cost': cost,
                            'cash_after': cash
                        })
                
                # 卖出信号
                elif signal == -1 and stock_code in positions:
                    pos = positions[stock_code]
                    shares = pos['shares']
                    entry_cost = pos['cost']
                    
                    # 考虑滑点
                    actual_price = price * (1 - self.slippage)
                    
                    # 计算卖出所得
                    proceeds = shares * actual_price
                    commission = proceeds * self.commission_rate
                    stamp = proceeds * self.stamp_tax  # 印花税
                    
                    net_proceeds = proceeds - commission - stamp
                    cash += net_proceeds
                    
                    # 计算盈亏
                    profit = net_proceeds - shares * entry_cost
                    profit_pct = profit / (shares * entry_cost)
                    
                    trade_log.append({
                        'date': date,
                        'stock_code': stock_code,
                        'action': 'SELL',
                        'shares': shares,
                        'price': actual_price,
                        'proceeds': net_proceeds,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'cash_after': cash
                    })
                    
                    del positions[stock_code]
            
            # 计算当日权益
            position_value = sum(
                pos['shares'] * day_data[day_data['stock_code'] == code]['trade_price'].iloc[0]
                if len(day_data[day_data['stock_code'] == code]) > 0 else 0
                for code, pos in positions.items()
            )
            
            total_equity = cash + position_value
            equity_curve.append({
                'date': date,
                'cash': cash,
                'position_value': position_value,
                'total_equity': total_equity
            })
            
            # 计算日收益率
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append(daily_return)
            prev_equity = total_equity
        
        # 整理结果
        equity_df = pd.DataFrame(equity_curve)
        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        
        return {
            'equity_curve': equity_df,
            'trade_log': trade_df,
            'daily_returns': np.array(daily_returns),
            'final_equity': equity_df['total_equity'].iloc[-1] if len(equity_df) > 0 else self.initial_capital,
            'final_cash': cash,
            'final_positions': positions
        }
    
    def calculate_metrics(self, backtest_results: Dict) -> Dict:
        """
        计算绩效指标
        
        计算的指标:
        1. 年化收益率
        2. 最大回撤
        3. 夏普比率
        4. 胜率
        5. 盈亏比
        6. 总交易次数
        """
        equity_curve = backtest_results['equity_curve']
        trade_log = backtest_results.get('trade_log', pd.DataFrame())
        daily_returns = backtest_results['daily_returns']
        
        if len(equity_curve) == 0:
            return self._empty_metrics()
        
        # 总收益率
        total_return = (equity_curve['total_equity'].iloc[-1] / self.initial_capital) - 1
        
        # 年化收益率
        n_days = len(equity_curve)
        annual_return = (1 + total_return) ** (250 / n_days) - 1 if n_days > 0 else 0
        
        # 最大回撤
        equity = equity_curve['total_equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 夏普比率
        risk_free_rate = 0.0275  # 无风险利率 2.75%
        excess_returns = daily_returns - risk_free_rate / 250
        sharpe_ratio = np.sqrt(250) * np.mean(excess_returns) / (np.std(daily_returns) + 1e-10)
        
        # 交易统计
        if len(trade_log) > 0 and 'action' in trade_log.columns:
            sell_trades = trade_log[trade_log['action'] == 'SELL']
            
            if len(sell_trades) > 0:
                winning_trades = sell_trades[sell_trades['profit'] > 0]
                losing_trades = sell_trades[sell_trades['profit'] < 0]
                
                win_rate = len(winning_trades) / len(sell_trades)
                
                avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 1
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                
                total_trades = len(sell_trades)
            else:
                win_rate = 0
                profit_loss_ratio = 0
                total_trades = 0
        else:
            win_rate = 0
            profit_loss_ratio = 0
            total_trades = 0
        
        # 年化波动率
        annual_volatility = np.std(daily_returns) * np.sqrt(250) if len(daily_returns) > 0 else 0
        
        # 索提诺比率（下行风险调整后收益）
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(250) if len(downside_returns) > 0 else 1
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'annual_volatility': annual_volatility
        }
    
    def _empty_metrics(self) -> Dict:
        """返回空的绩效指标"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'total_trades': 0,
            'annual_volatility': 0
        }
    
    def buy_and_hold(self, price_data: pd.DataFrame) -> Dict:
        """
        买入持有策略（向量化实现，高性能）
        
        在期初等权重买入所有股票，持有至期末
        """
        stocks = price_data['stock_code'].unique()
        n_stocks = len(stocks)
        
        if n_stocks == 0:
            return {'equity_curve': pd.DataFrame(), 'daily_returns': np.array([])}
        
        capital_per_stock = self.initial_capital / n_stocks
        
        # 向量化：pivot为日期×股票的收盘价矩阵
        pivot = price_data.pivot_table(index='date', columns='stock_code', values='close')
        pivot = pivot.sort_index()
        
        # 每只股票在第一天买入的股数（取整到100股）
        initial_prices = pivot.iloc[0].dropna()
        pivot = pivot[initial_prices.index]  # 只保留首日有数据的股票
        shares = (capital_per_stock / initial_prices / 100).astype(int) * 100
        
        # 每日总市值 = 各股票 shares × 当日收盘价（NaN用前值填充）
        pivot = pivot.ffill()
        daily_value = (pivot * shares).sum(axis=1)
        # 未买到的剩余现金
        cash_remainder = self.initial_capital - (shares * initial_prices).sum()
        daily_value += cash_remainder
        
        equity_df = pd.DataFrame({
            'date': daily_value.index,
            'total_equity': daily_value.values
        }).reset_index(drop=True)
        
        equity_df['daily_return'] = equity_df['total_equity'].pct_change().fillna(0)
        
        return {
            'equity_curve': equity_df,
            'daily_returns': equity_df['daily_return'].values,
            'trade_log': pd.DataFrame()
        }
    
    def ma_strategy(self, price_data: pd.DataFrame,
                   short_window: int = 5,
                   long_window: int = 20) -> Dict:
        """
        传统均线策略
        
        MA5上穿MA20买入，下穿卖出
        """
        results = []
        
        for stock in price_data['stock_code'].unique():
            stock_df = price_data[price_data['stock_code'] == stock].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # 计算均线
            stock_df['ma_short'] = stock_df['close'].rolling(short_window).mean()
            stock_df['ma_long'] = stock_df['close'].rolling(long_window).mean()
            
            # 生成信号
            stock_df['signal'] = 0
            stock_df.loc[stock_df['ma_short'] > stock_df['ma_long'], 'signal'] = 1
            stock_df.loc[stock_df['ma_short'] <= stock_df['ma_long'], 'signal'] = -1
            
            # 信号变化时触发交易
            stock_df['trade_signal'] = stock_df['signal'].diff()
            stock_df.loc[stock_df['trade_signal'] > 0, 'trade_signal'] = 1    # 买入
            stock_df.loc[stock_df['trade_signal'] < 0, 'trade_signal'] = -1   # 卖出
            stock_df['trade_signal'] = stock_df['trade_signal'].fillna(0)
            
            results.append(stock_df)
        
        combined_df = pd.concat(results, ignore_index=True)
        
        # 添加position_size列
        combined_df['position_size'] = 0.2
        combined_df['signal'] = combined_df['trade_signal'].astype(int)
        
        # 使用回测引擎
        return self.run_backtest(
            signals=combined_df[['date', 'stock_code', 'signal', 'position_size']],
            price_data=price_data
        )


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=250, freq='B')
    
    price_data = pd.DataFrame({
        'date': dates,
        'stock_code': ['600519.SH'] * 250,
        'close': 100 * np.cumprod(1 + np.random.randn(250) * 0.02),
        'open': 100 * np.cumprod(1 + np.random.randn(250) * 0.02)
    })
    
    signals = pd.DataFrame({
        'date': dates,
        'stock_code': ['600519.SH'] * 250,
        'signal': np.random.choice([-1, 0, 1], 250, p=[0.2, 0.6, 0.2]),
        'position_size': 0.2
    })
    
    # 回测
    engine = BacktestEngine(initial_capital=1000000)
    
    print("=" * 50)
    print("策略回测测试")
    print("=" * 50)
    
    # 本文策略
    results = engine.run_backtest(signals, price_data)
    metrics = engine.calculate_metrics(results)
    
    print("\n本文策略绩效:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 买入持有策略
    bh_results = engine.buy_and_hold(price_data)
    bh_metrics = engine.calculate_metrics(bh_results)
    
    print("\n买入持有策略绩效:")
    print(f"  年化收益率: {bh_metrics['annual_return']:.4f}")
    print(f"  最大回撤: {bh_metrics['max_drawdown']:.4f}")
    
    # 均线策略
    ma_results = engine.ma_strategy(price_data)
    ma_metrics = engine.calculate_metrics(ma_results)
    
    print("\n均线策略绩效:")
    print(f"  年化收益率: {ma_metrics['annual_return']:.4f}")
    print(f"  最大回撤: {ma_metrics['max_drawdown']:.4f}")
