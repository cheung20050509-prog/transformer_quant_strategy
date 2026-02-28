# -*- coding: utf-8 -*-
"""
特征工程模块
============
功能：
1. 计算技术指标（MACD、RSI、布林带、均线等）
2. 构建时序特征
3. 特征标准化
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self):
        """初始化"""
        self.feature_columns = []
        self.scalers = {}
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标特征
        
        参数:
            df: 原始数据DataFrame
        
        返回:
            DataFrame: 包含所有特征的DataFrame
        """
        df = df.copy()
        
        print("  计算技术指标特征...")
        
        # 按股票分组计算特征
        result_dfs = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # 1. 移动平均线
            stock_df = self._compute_ma(stock_df)
            
            # 2. MACD指标
            stock_df = self._compute_macd(stock_df)
            
            # 3. RSI指标
            stock_df = self._compute_rsi(stock_df)
            
            # 4. 布林带
            stock_df = self._compute_bollinger(stock_df)
            
            # 5. ATR波动率指标
            stock_df = self._compute_atr(stock_df)
            
            # 6. 动量指标
            stock_df = self._compute_momentum(stock_df)
            
            # 7. 成交量特征
            stock_df = self._compute_volume_features(stock_df)
            
            # 8. 价格特征
            stock_df = self._compute_price_features(stock_df)
            
            # 9. 七大王牌指标信号（来自《七大王牌指标必杀技》）
            stock_df = self._compute_seven_signals(stock_df)
            
            # 10. KDJ指标
            stock_df = self._compute_kdj(stock_df)
            
            # 11. DMI指标
            stock_df = self._compute_dmi(stock_df)
            
            # 12. 计算未来收益率（预测目标）
            stock_df = self._compute_future_returns(stock_df)
            
            result_dfs.append(stock_df)
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        # 处理无穷值和过大值
        for col in result_df.select_dtypes(include=[np.number]).columns:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
            result_df[col] = result_df[col].fillna(result_df[col].median())
        
        # 获取特征列
        feature_cols = self.get_feature_columns()
        target_cols = ['future_return_1d', 'future_return_5d', 'future_direction']
        required_cols = feature_cols + target_cols
        
        # 只检查必要列的NaN（忽略其他列的NaN）
        available_cols = [col for col in required_cols if col in result_df.columns]
        initial_count = len(result_df)
        
        # 填充一些可能全为NaN或0的列
        for col in ['turnover_change', 'vol_price_corr']:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)
        
        result_df = result_df.dropna(subset=available_cols)
        final_count = len(result_df)
        
        print(f"  特征计算完成: 原始 {initial_count} 条 → 有效 {final_count} 条")
        print(f"  特征数量: {len(self.get_feature_columns())}")
        
        return result_df
    
    def _compute_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 均线偏离度
        df['ma5_bias'] = (df['close'] - df['ma5']) / df['ma5']
        df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
        
        # 均线趋势（短期均线与长期均线的关系）
        df['ma_trend'] = (df['ma5'] - df['ma20']) / df['ma20']
        
        return df
    
    def _compute_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """
        计算MACD指标
        
        MACD = DIF - DEA
        DIF = EMA(12) - EMA(26)
        DEA = EMA(DIF, 9)
        """
        # 计算EMA
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # DIF线
        df['macd_dif'] = ema_fast - ema_slow
        
        # DEA线（信号线）
        df['macd_dea'] = df['macd_dif'].ewm(span=signal, adjust=False).mean()
        
        # MACD柱状图
        df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        # MACD金叉/死叉信号
        df['macd_cross'] = np.where(df['macd_dif'] > df['macd_dea'], 1, -1)
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, periods=[6, 12, 24]) -> pd.DataFrame:
        """
        计算RSI指标
        
        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度
        """
        delta = df['close'].diff()
        
        for period in periods:
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI超买超卖信号
        df['rsi_signal'] = np.where(df['rsi_12'] > 70, -1, 
                                     np.where(df['rsi_12'] < 30, 1, 0))
        
        return df
    
    def _compute_bollinger(self, df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
        """
        计算布林带
        
        中轨 = MA20
        上轨 = MA20 + 2σ
        下轨 = MA20 - 2σ
        """
        df['boll_mid'] = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        df['boll_upper'] = df['boll_mid'] + num_std * rolling_std
        df['boll_lower'] = df['boll_mid'] - num_std * rolling_std
        
        # 布林带宽度
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
        
        # 价格在布林带中的位置 (0-1)
        df['boll_pct'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-10)
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """
        计算ATR（平均真实波幅）
        
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = MA(TR, period)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        
        # ATR百分比（相对于收盘价）
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        # 价格动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # ROC (Rate of Change)
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # 价格加速度
        df['price_accel'] = df['momentum_5'] - df['momentum_5'].shift(5)
        
        return df
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量特征"""
        # 成交量均线
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        
        # 成交量比率
        df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)
        
        # 量价关系
        df['vol_price_corr'] = df['close'].rolling(window=10).corr(df['volume'])
        
        # 换手率变化（处理全为0的情况）
        if df['turnover_rate'].abs().sum() > 0:
            df['turnover_change'] = df['turnover_rate'].pct_change()
        else:
            df['turnover_change'] = 0.0
        
        return df
    
    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格特征"""
        # 日内波动
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        
        # 收盘位置（在日内高低点之间的位置）
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # 跳空缺口
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 日收益率
        df['return_1d'] = df['close'].pct_change()
        
        # 历史波动率
        df['volatility_20'] = df['return_1d'].rolling(window=20).std() * np.sqrt(250)
        
        return df
    
    def _compute_seven_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算七大王牌指标信号（参考《七大王牌指标必杀技》）
        
        包括:
        1. MACD多空信号（烽火连天、黄金交叉、死亡交叉）
        2. 均线多空排列
        3. K线形态信号
        4. 布林带突破信号
        5. 成交量信号
        """
        # === 1. MACD高级信号 ===
        # MACD零轴上下
        df['macd_above_zero'] = (df['macd_dif'] > 0).astype(int)
        
        # MACD金叉条件：DIF上穿DEA
        df['macd_golden_cross'] = ((df['macd_dif'] > df['macd_dea']) & 
                                    (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))).astype(int)
        
        # MACD死叉条件：DIF下穿DEA
        df['macd_death_cross'] = ((df['macd_dif'] < df['macd_dea']) & 
                                   (df['macd_dif'].shift(1) >= df['macd_dea'].shift(1))).astype(int)
        
        # 零轴上金叉（强势做多信号）
        df['macd_strong_buy'] = ((df['macd_golden_cross'] == 1) & (df['macd_dif'] > 0)).astype(int)
        
        # 零轴下死叉（强势做空信号）
        df['macd_strong_sell'] = ((df['macd_death_cross'] == 1) & (df['macd_dif'] < 0)).astype(int)
        
        # MACD柱状图变化趋势
        df['macd_hist_trend'] = df['macd_hist'] - df['macd_hist'].shift(1)
        
        # 红柱子放大（多头加力）
        df['macd_hist_expanding'] = ((df['macd_hist'] > 0) & (df['macd_hist_trend'] > 0)).astype(int)
        
        # 绿柱子缩小（空头衰竭）
        df['macd_hist_shrinking'] = ((df['macd_hist'] < 0) & (df['macd_hist_trend'] > 0)).astype(int)
        
        # === 2. 均线多空排列信号 ===
        # 多头排列: MA5 > MA10 > MA20 > MA60
        df['ma_bullish_align'] = ((df['ma5'] > df['ma10']) & 
                                   (df['ma10'] > df['ma20']) & 
                                   (df['ma20'] > df['ma60'])).astype(int)
        
        # 空头排列: MA5 < MA10 < MA20 < MA60
        df['ma_bearish_align'] = ((df['ma5'] < df['ma10']) & 
                                   (df['ma10'] < df['ma20']) & 
                                   (df['ma20'] < df['ma60'])).astype(int)
        
        # 均线粘合（震荡市场）
        ma_range = (df[['ma5', 'ma10', 'ma20']].max(axis=1) - 
                    df[['ma5', 'ma10', 'ma20']].min(axis=1)) / df['close']
        df['ma_convergence'] = (ma_range < 0.02).astype(int)
        
        # 价格站上/跌破60日均线
        df['price_above_ma60'] = (df['close'] > df['ma60']).astype(int)
        
        # === 3. K线形态信号 ===
        # 大阳线（涨幅>5%）
        df['big_yang'] = (df['pct_change'] > 5).astype(int)
        
        # 大阴线（跌幅>5%）
        df['big_yin'] = (df['pct_change'] < -5).astype(int)
        
        # 十字星：实体小于振幅的30%
        body = abs(df['close'] - df['open'])
        amplitude = df['high'] - df['low']
        df['doji'] = (body < amplitude * 0.3).astype(int)
        
        # 锤子线（下影线长，实体短）
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        df['hammer'] = ((lower_shadow > body * 2) & 
                        (upper_shadow < body * 0.5) & 
                        (df['pct_change'] > 0)).astype(int)
        
        # 倒锤子线
        df['inv_hammer'] = ((upper_shadow > body * 2) & 
                            (lower_shadow < body * 0.5) & 
                            (df['pct_change'] < 0)).astype(int)
        
        # 好友反攻形态（简化版）
        df['bullish_reversal'] = ((df['pct_change'].shift(1) < -2) & 
                                   (df['pct_change'] > 2)).astype(int)
        
        # 淡友反攻形态
        df['bearish_reversal'] = ((df['pct_change'].shift(1) > 2) & 
                                   (df['pct_change'] < -2)).astype(int)
        
        # === 4. 布林带突破信号 ===
        # 突破上轨
        df['boll_break_upper'] = ((df['close'] > df['boll_upper']) & 
                                   (df['close'].shift(1) <= df['boll_upper'].shift(1))).astype(int)
        
        # 跌破下轨
        df['boll_break_lower'] = ((df['close'] < df['boll_lower']) & 
                                   (df['close'].shift(1) >= df['boll_lower'].shift(1))).astype(int)
        
        # 布林带收窄（即将突破）
        boll_width_ma = df['boll_width'].rolling(20).mean()
        df['boll_squeeze'] = (df['boll_width'] < boll_width_ma * 0.7).astype(int)
        
        # === 5. 成交量信号 ===
        # 放量上涨（量价齐升）
        df['vol_price_up'] = ((df['vol_ratio'] > 1.5) & (df['pct_change'] > 0)).astype(int)
        
        # 放量下跌
        df['vol_price_down'] = ((df['vol_ratio'] > 1.5) & (df['pct_change'] < 0)).astype(int)
        
        # 缩量调整
        df['vol_shrink'] = (df['vol_ratio'] < 0.6).astype(int)
        
        # 天量（异常放量，可能见顶）
        df['extreme_volume'] = (df['vol_ratio'] > 3).astype(int)
        
        # === 6. 综合信号强度 ===
        # 多头信号强度
        df['bull_signal_strength'] = (
            df['macd_strong_buy'] * 2 +
            df['macd_hist_expanding'] +
            df['ma_bullish_align'] * 2 +
            df['hammer'] +
            df['bullish_reversal'] * 2 +
            df['vol_price_up'] +
            (df['rsi_12'] < 30).astype(int) * 2  # RSI超卖
        )
        
        # 空头信号强度
        df['bear_signal_strength'] = (
            df['macd_strong_sell'] * 2 +
            (df['macd_hist'] < 0).astype(int) +
            df['ma_bearish_align'] * 2 +
            df['inv_hammer'] +
            df['bearish_reversal'] * 2 +
            df['vol_price_down'] +
            (df['rsi_12'] > 70).astype(int) * 2  # RSI超买
        )
        
        # 净信号强度
        df['net_signal'] = df['bull_signal_strength'] - df['bear_signal_strength']
        
        return df
    
    def _compute_kdj(self, df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标
        
        K = MA(RSV, m1)
        D = MA(K, m2)
        J = 3*K - 2*D
        
        RSV = (Close - LLV(Low, n)) / (HHV(High, n) - LLV(Low, n)) * 100
        """
        # 计算RSV
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-10) * 100
        
        # 计算K, D, J（使用EMA）
        df['kdj_k'] = rsv.ewm(span=m1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(span=m2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # KDJ金叉
        df['kdj_golden_cross'] = ((df['kdj_k'] > df['kdj_d']) & 
                                   (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))).astype(int)
        
        # KDJ死叉
        df['kdj_death_cross'] = ((df['kdj_k'] < df['kdj_d']) & 
                                  (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))).astype(int)
        
        # KDJ超买超卖
        df['kdj_overbought'] = (df['kdj_j'] > 100).astype(int)
        df['kdj_oversold'] = (df['kdj_j'] < 0).astype(int)
        
        return df
    
    def _compute_dmi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算DMI指标（动向指标）
        
        +DM = High - Previous High (if > 0 and > -DM)
        -DM = Previous Low - Low (if > 0 and > +DM)
        TR = max(H-L, |H-Prev C|, |L-Prev C|)
        +DI = EMA(+DM, period) / EMA(TR, period) * 100
        -DI = EMA(-DM, period) / EMA(TR, period) * 100
        ADX = EMA(|+DI - -DI| / (+DI + -DI) * 100, period)
        """
        # 计算+DM和-DM
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # 计算TR
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - prev_close)
        tr3 = abs(df['low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算+DI和-DI
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        df['dmi_plus_di'] = plus_dm_smooth / (atr + 1e-10) * 100
        df['dmi_minus_di'] = minus_dm_smooth / (atr + 1e-10) * 100
        
        # 计算ADX
        dx = abs(df['dmi_plus_di'] - df['dmi_minus_di']) / (df['dmi_plus_di'] + df['dmi_minus_di'] + 1e-10) * 100
        df['dmi_adx'] = dx.ewm(span=period, adjust=False).mean()
        
        # DMI多空信号
        df['dmi_bullish'] = (df['dmi_plus_di'] > df['dmi_minus_di']).astype(int)
        
        # ADX趋势强度
        df['dmi_strong_trend'] = (df['dmi_adx'] > 25).astype(int)
        
        return df
    
    def _compute_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算未来收益率（预测目标）"""
        # 未来1日收益率
        df['future_return_1d'] = df['close'].shift(-1) / df['close'] - 1
        
        # 未来3日收益率
        df['future_return_3d'] = df['close'].shift(-3) / df['close'] - 1
        
        # 未来5日收益率
        df['future_return_5d'] = df['close'].shift(-5) / df['close'] - 1
        
        # 未来收益方向（分类目标）
        df['future_direction'] = np.where(df['future_return_5d'] > 0, 1, 0)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        self.feature_columns = [
            # 均线特征
            'ma5', 'ma10', 'ma20', 'ma60', 'ma5_bias', 'ma20_bias', 'ma_trend',
            # MACD特征
            'macd_dif', 'macd_dea', 'macd_hist', 'macd_cross',
            # RSI特征
            'rsi_6', 'rsi_12', 'rsi_24', 'rsi_signal',
            # 布林带特征
            'boll_mid', 'boll_upper', 'boll_lower', 'boll_width', 'boll_pct',
            # ATR特征
            'atr', 'atr_pct',
            # 动量特征
            'momentum_5', 'momentum_10', 'momentum_20', 'roc_10', 'price_accel',
            # 成交量特征
            'vol_ma5', 'vol_ma20', 'vol_ratio', 'vol_price_corr', 'turnover_change',
            # 价格特征
            'intraday_range', 'close_position', 'gap', 'return_1d', 'volatility_20',
            # 七大王牌MACD信号
            'macd_above_zero', 'macd_golden_cross', 'macd_death_cross',
            'macd_strong_buy', 'macd_strong_sell', 'macd_hist_trend',
            'macd_hist_expanding', 'macd_hist_shrinking',
            # 均线排列信号
            'ma_bullish_align', 'ma_bearish_align', 'ma_convergence', 'price_above_ma60',
            # K线形态信号
            'big_yang', 'big_yin', 'doji', 'hammer', 'inv_hammer',
            'bullish_reversal', 'bearish_reversal',
            # 布林带信号
            'boll_break_upper', 'boll_break_lower', 'boll_squeeze',
            # 成交量信号
            'vol_price_up', 'vol_price_down', 'vol_shrink', 'extreme_volume',
            # 综合信号
            'bull_signal_strength', 'bear_signal_strength', 'net_signal',
            # KDJ指标
            'kdj_k', 'kdj_d', 'kdj_j', 'kdj_golden_cross', 'kdj_death_cross',
            'kdj_overbought', 'kdj_oversold',
            # DMI指标
            'dmi_plus_di', 'dmi_minus_di', 'dmi_adx', 'dmi_bullish', 'dmi_strong_trend',
            # 基础特征
            'close', 'volume', 'turnover_rate'
        ]
        return self.feature_columns
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str], 
                          method: str = 'zscore') -> Tuple[pd.DataFrame, dict]:
        """
        特征标准化
        
        参数:
            df: 特征数据
            feature_cols: 需要标准化的特征列
            method: 标准化方法 ('zscore' 或 'minmax')
        
        返回:
            Tuple[DataFrame, dict]: 标准化后的数据和scaler参数
        """
        df = df.copy()
        scalers = {}
        
        for col in feature_cols:
            if col in df.columns:
                if method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = (df[col] - mean) / (std + 1e-10)
                    scalers[col] = {'mean': mean, 'std': std}
                elif method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
                    scalers[col] = {'min': min_val, 'max': max_val}
        
        self.scalers = scalers
        return df, scalers


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    from data_acquisition import DataAcquisition
    
    da = DataAcquisition()
    raw_data = da.fetch_stock_data(
        stock_list=['600519.SH'],
        start_date='20230101',
        end_date='20251231'
    )
    cleaned_data = da.clean_data(raw_data)
    
    # 特征工程
    fe = FeatureEngineering()
    feature_data = fe.compute_all_features(cleaned_data)
    
    print("\n特征列表:")
    for col in fe.get_feature_columns():
        print(f"  - {col}")
    
    print(f"\n特征数据形状: {feature_data.shape}")
    print(feature_data.head())
