# -*- coding: utf-8 -*-
"""
数据获取与清洗模块
==================
功能：
1. 从Tushare获取A股股票日度数据
2. 使用SQLite存储数据便于复用
3. 数据清洗（缺失值处理、异常值处理）
4. 数据描述性统计
"""

import pandas as pd
import numpy as np
import sqlite3
import tushare as ts
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Tushare Token（从ecofinal项目获取）
TUSHARE_TOKEN = '229e2c478deaef0ccf3030b42121cc7b5ba066dd3c9789b4835c943d'

# 默认数据库路径
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')


class DataAcquisition:
    """数据获取与清洗类"""
    
    def __init__(self, db_path: str = None):
        """
        初始化
        
        参数:
            db_path: SQLite数据库路径，默认为当前目录下的stock_data.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.raw_data = None
        self.cleaned_data = None
        
        # 初始化Tushare
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        print(f"  Tushare已初始化，数据库路径: {self.db_path}")
    
    def _get_db_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def _table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def _load_from_db(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从SQLite数据库加载数据
        
        返回:
            DataFrame 或 None（如果数据不存在或不完整）
        """
        table_name = f"stock_{stock_code.replace('.', '_')}"
        
        if not self._table_exists(table_name):
            return None
        
        conn = self._get_db_connection()
        try:
            # 读取指定日期范围的数据
            query = f"""
                SELECT * FROM {table_name} 
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            df = pd.read_sql(query, conn)
            
            if len(df) == 0:
                return None
            
            # 检查数据完整性（至少需要覆盖80%的交易日）
            expected_days = pd.bdate_range(
                start=datetime.strptime(start_date, '%Y%m%d'),
                end=datetime.strptime(end_date, '%Y%m%d')
            )
            if len(df) < len(expected_days) * 0.8:
                return None
            
            return df
            
        except Exception as e:
            print(f"    从数据库加载失败: {e}")
            return None
        finally:
            conn.close()
    
    def _save_to_db(self, df: pd.DataFrame, stock_code: str):
        """保存数据到SQLite数据库"""
        table_name = f"stock_{stock_code.replace('.', '_')}"
        conn = self._get_db_connection()
        try:
            # 如果表存在，先删除旧数据再插入新数据
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"    ✓ 数据已保存到 {table_name} 表")
        except Exception as e:
            print(f"    保存到数据库失败: {e}")
        finally:
            conn.close()
    
    def fetch_stock_data(self, stock_list: list, start_date: str, end_date: str,
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        获取股票日度数据
        
        参数:
            stock_list: 股票代码列表，如 ['600519.SH', '000858.SZ']
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            force_refresh: 是否强制从Tushare重新获取数据
        
        返回:
            DataFrame: 包含所有股票数据的合并DataFrame
        """
        all_data = []
        
        for stock_code in stock_list:
            print(f"  正在处理 {stock_code}...")
            
            # 1. 尝试从数据库加载
            if not force_refresh:
                df = self._load_from_db(stock_code, start_date, end_date)
                if df is not None:
                    df = self._standardize_columns(df, stock_code)
                    all_data.append(df)
                    print(f"    ✓ 从数据库加载成功，共 {len(df)} 条记录")
                    continue
            
            # 2. 从Tushare获取数据
            try:
                df = self._fetch_from_tushare(stock_code, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    # 保存到数据库
                    self._save_to_db(df, stock_code)
                    
                    # 标准化列名
                    df = self._standardize_columns(df, stock_code)
                    all_data.append(df)
                    print(f"    ✓ 从Tushare获取成功，共 {len(df)} 条记录")
                else:
                    print(f"    ✗ {stock_code} 数据为空")
                    
            except Exception as e:
                print(f"    ✗ 获取 {stock_code} 失败: {str(e)}")
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            print(f"\n  数据获取完成，总计 {len(self.raw_data)} 条记录")
            return self.raw_data
        else:
            raise ValueError("未能获取任何股票数据")
    
    def _fetch_from_tushare(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从Tushare获取股票日度数据
        """
        # Tushare使用的股票代码格式: 600519.SH
        df = self.pro.daily(
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or len(df) == 0:
            return None
        
        # 获取复权因子并计算前复权价格
        adj_df = self.pro.adj_factor(
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if adj_df is not None and len(adj_df) > 0:
            df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
            # 前复权计算
            latest_adj = adj_df['adj_factor'].iloc[0]  # 最新的复权因子
            df['adj_ratio'] = df['adj_factor'] / latest_adj
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * df['adj_ratio']
        
        # 按日期升序排列
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化列名，确保与后续处理一致
        """
        df = df.copy()
        
        # 列名映射
        column_mapping = {
            'trade_date': 'date',
            'ts_code': 'stock_code',
            'vol': 'volume',
            'pct_chg': 'pct_change',
            'pre_close': 'pre_close',
            'change': 'change',
            'amount': 'amount',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保有stock_code列
        if 'stock_code' not in df.columns:
            df['stock_code'] = stock_code
        
        # 日期转换
        df['date'] = pd.to_datetime(df['date'])
        
        # 添加换手率（如果没有）
        if 'turnover_rate' not in df.columns:
            df['turnover_rate'] = 0.0
        
        # 添加振幅（如果没有）
        if 'amplitude' not in df.columns:
            df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
        
        # 选择需要的列
        required_cols = ['date', 'stock_code', 'open', 'high', 'low', 'close', 
                         'volume', 'amount', 'pct_change', 'change', 'turnover_rate', 'amplitude']
        
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def clean_data(self, df: pd.DataFrame, advanced_clean: bool = True) -> pd.DataFrame:
        """
        数据清洗（增强版）
        
        步骤:
        1. 处理缺失值（前向填充 + 插值）
        2. 处理异常值（IQR方法 + 3σ原则结合）
        3. 检测并标记停牌日
        4. 过滤无效交易日（成交量为0）
        5. 价格一致性检查
        6. 确保数据类型正确
        7. 按日期排序
        """
        print("\n  数据清洗步骤（增强版）:")
        
        df = df.copy()
        initial_count = len(df)
        
        # 1. 检查并处理缺失值
        missing_before = df.isnull().sum().sum()
        print(f"  [1] 缺失值数量: {missing_before}")
        
        # 数值列
        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 
                        'amplitude', 'pct_change', 'change', 'turnover_rate']
        
        for col in numeric_cols:
            if col in df.columns:
                # 先用前向填充
                df[col] = df.groupby('stock_code')[col].ffill()
                # 再用后向填充处理开头的缺失值
                df[col] = df.groupby('stock_code')[col].bfill()
                # 对于仍有缺失的，用线性插值
                df[col] = df.groupby('stock_code')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
        
        missing_after = df.isnull().sum().sum()
        print(f"      处理后缺失值: {missing_after}")
        
        # 2. 检测并过滤停牌日（成交量为0的交易日）
        if 'volume' in df.columns:
            suspension_days = df[df['volume'] == 0]
            if len(suspension_days) > 0:
                print(f"  [2] 检测到 {len(suspension_days)} 个停牌日，已过滤")
                df = df[df['volume'] > 0]
        
        # 3. 价格一致性检查
        price_errors = 0
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 检查 high >= low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                price_errors += invalid_hl.sum()
                # 交换 high 和 low
                df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
            
            # 检查 high >= open, close 和 low <= open, close
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        print(f"  [3] 价格一致性检查: 修正 {price_errors} 条记录")
        
        # 4. 异常值处理（IQR + 3σ结合方法）
        outliers_count = 0
        outlier_cols = ['volume', 'turnover_rate', 'pct_change', 'amplitude']
        
        if advanced_clean:
            for col in outlier_cols:
                if col in df.columns:
                    # 使用IQR方法
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # 3σ方法
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std > 0 and IQR > 0:
                        # 结合两种方法，取更宽松的范围
                        lower_bound = max(Q1 - 3 * IQR, mean - 4 * std)
                        upper_bound = min(Q3 + 3 * IQR, mean + 4 * std)
                        
                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                        outliers_count += outliers.sum()
                        
                        # 用分位数替换异常值（比中位数更稳健）
                        df.loc[df[col] < lower_bound, col] = Q1
                        df.loc[df[col] > upper_bound, col] = Q3
        
        print(f"  [4] 异常值处理: 共处理 {outliers_count} 个异常值")
        
        # 5. 涨跌幅限制检查（A股±10%，ST股±5%）
        if 'pct_change' in df.columns:
            extreme_changes = (df['pct_change'].abs() > 11)  # 允许一点误差
            if extreme_changes.any():
                print(f"  [5] 发现 {extreme_changes.sum()} 条涨跌幅异常记录，已修正")
                df.loc[extreme_changes, 'pct_change'] = np.clip(
                    df.loc[extreme_changes, 'pct_change'], -10, 10
                )
        
        # 6. 确保数据类型正确
        df['date'] = pd.to_datetime(df['date'])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"  [6] 数据类型转换完成")
        
        # 7. 按股票代码和日期排序
        df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
        print(f"  [7] 数据排序完成")
        
        # 8. 删除仍有缺失值的行
        df = df.dropna()
        final_count = len(df)
        
        print(f"\n  清洗完成: {initial_count} → {final_count} 条记录")
        print(f"  数据保留率: {final_count/initial_count*100:.2f}%")
        
        self.cleaned_data = df
        return df
    
    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算描述性统计
        """
        # 按股票分组计算统计量
        stats_list = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code]
            
            stats = {
                '股票代码': stock_code,
                '数据条数': len(stock_df),
                '起始日期': stock_df['date'].min().strftime('%Y-%m-%d'),
                '结束日期': stock_df['date'].max().strftime('%Y-%m-%d'),
                '平均收盘价': stock_df['close'].mean(),
                '收盘价标准差': stock_df['close'].std(),
                '平均日收益率(%)': stock_df['pct_change'].mean(),
                '收益率标准差(%)': stock_df['pct_change'].std(),
                '平均成交量': stock_df['volume'].mean(),
                '平均换手率(%)': stock_df['turnover_rate'].mean(),
                '最大涨幅(%)': stock_df['pct_change'].max(),
                '最大跌幅(%)': stock_df['pct_change'].min(),
            }
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        
        # 打印汇总
        print("\n  数据描述性统计汇总:")
        print(f"  - 股票数量: {len(stats_df)}")
        print(f"  - 总数据条数: {df.shape[0]}")
        print(f"  - 平均日收익率: {df['pct_change'].mean():.4f}%")
        print(f"  - 收益率标准差: {df['pct_change'].std():.4f}%")
        
        return stats_df
    
    def list_cached_stocks(self) -> list:
        """列出数据库中已缓存的股票列表"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stock_%'")
        tables = cursor.fetchall()
        conn.close()
        
        stocks = []
        for table in tables:
            # 从表名恢复股票代码: stock_600519_SH -> 600519.SH
            parts = table[0].replace('stock_', '').rsplit('_', 1)
            if len(parts) == 2:
                stocks.append(f"{parts[0]}.{parts[1]}")
        
        return stocks


# 测试代码
if __name__ == '__main__':
    # 测试数据获取
    da = DataAcquisition()
    
    # 沪深300代表性股票
    stock_list = [
        '600519.SH',  # 贵州茅台
        '000858.SZ',  # 五粮液
        '601318.SH',  # 中国平安
        '300750.SZ',  # 宁德时代
        '002594.SZ',  # 比亚迪
    ]
    
    raw_data = da.fetch_stock_data(
        stock_list=stock_list,
        start_date='20200101',
        end_date='20260131'
    )
    
    cleaned_data = da.clean_data(raw_data)
    
    stats = da.descriptive_statistics(cleaned_data)
    print("\n统计结果:")
    print(stats)
