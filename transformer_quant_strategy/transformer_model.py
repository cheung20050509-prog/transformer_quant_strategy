# -*- coding: utf-8 -*-
"""
iTransformer模型模块 (2024 ICLR)
================================
功能：
1. 实现iTransformer时序预测模型 - 反转注意力机制
2. 多股票关联建模 - 股票间attention学习联动关系
3. 模型训练与预测
4. 模型评估

核心创新（iTransformer vs 传统Transformer）：
- 传统Transformer: 在时间维度做attention (时间步之间)
- iTransformer: 在变量/股票维度做attention (股票之间)
- 优势: 更好捕捉多变量/多股票的关联关系

模型结构：
- 输入: 多只股票过去N天的特征序列 (batch, n_stocks, seq_len, features)
- 时间编码: 每只股票独立编码时间序列
- 股票Attention: 在股票维度做cross-attention学习关联
- 输出: 各股票未来M天的收益率预测
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import hashlib
import json


def set_seed(seed: int = 42):
    """固定所有随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # DataLoader worker seed fix
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(f"  [随机种子] 已固定 seed={seed}")
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为序列数据添加位置信息，使Transformer能够感知序列顺序
    
    计算公式:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleConv(nn.Module):
    """
    多尺度卷积特征提取模块
    
    使用不同窗口大小的卷积核捕捉短期、中期、长期模式
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(MultiScaleConv, self).__init__()
        
        # 不同尺度的卷积
        self.conv1 = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, output_dim // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(input_dim, output_dim // 4, kernel_size=1)  # 点卷积
        
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        # 多尺度卷积
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        
        # 拼接 -> (batch, output_dim, seq)
        out = torch.cat([c1, c2, c3, c4], dim=1)
        
        # 转回 (batch, seq, output_dim)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.activation(out)
        
        return out


# ============================================================
# iTransformer 核心组件 (2024 ICLR)
# ============================================================

class InvertedAttention(nn.Module):
    """
    反转注意力模块 - iTransformer核心
    
    传统Transformer: Q,K,V来自不同时间步 -> 时间步间attention
    iTransformer: Q,K,V来自不同变量/股票 -> 变量/股票间attention
    
    这让模型能捕捉：
    - 茅台与五粮液的联动 (消费板块)
    - 平安与招行的关联 (金融板块)
    - 跨板块的领先滞后关系
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super(InvertedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # 存储attention权重用于可视化
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_vars, d_model) - n_vars是变量/股票数量
        Returns:
            (batch, n_vars, d_model)
        """
        batch_size, n_vars, _ = x.shape
        
        # 线性投影
        Q = self.q_proj(x)  # (batch, n_vars, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 多头分割
        Q = Q.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, n_vars, head_dim)
        
        # 计算attention分数 - 变量/股票之间
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (batch, n_heads, n_vars, n_vars) - 股票间关联矩阵!
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.attention_weights = attn_weights.detach()  # 保存用于可视化
        attn_weights = self.dropout(attn_weights)
        
        # 应用attention
        out = torch.matmul(attn_weights, V)  # (batch, n_heads, n_vars, head_dim)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, n_vars, self.d_model)
        out = self.out_proj(out)
        
        return out


class iTransformerBlock(nn.Module):
    """
    iTransformer单层模块
    
    结构: InvertedAttention -> FFN -> LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 256, dropout: float = 0.1):
        super(iTransformerBlock, self).__init__()
        
        self.attention = InvertedAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm架构
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    """
    时间序列编码器
    
    将每只股票的时间序列编码为固定维度表示
    使用1D卷积 + MLP进行时间聚合
    """
    def __init__(self, seq_len: int, input_dim: int, d_model: int, dropout: float = 0.1):
        super(TemporalEncoder, self).__init__()
        
        # 多尺度时间卷积
        self.conv1 = nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, d_model // 2, kernel_size=7, padding=3)
        
        # 时间聚合 - 将序列压缩为单个表示
        self.temporal_pool = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, 1)  # 聚合为单点
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - 单只股票的时间序列
        Returns:
            (batch, d_model) - 该股票的表示向量
        """
        # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 多尺度卷积
        c1 = self.conv1(x)  # (batch, d_model//2, seq_len)
        c2 = self.conv2(x)
        conv_out = torch.cat([c1, c2], dim=1)  # (batch, d_model, seq_len)
        
        # 时间聚合
        pooled = self.temporal_pool(conv_out).squeeze(-1)  # (batch, d_model)
        
        # LayerNorm + 投影
        out = self.proj(self.norm(pooled))
        
        return out


class iTransformerEncoder(nn.Module):
    """
    iTransformer完整编码器
    
    流程:
    1. 每只股票独立进行时间编码 -> (batch, n_stocks, d_model)
    2. 在股票维度做self-attention学习关联 -> (batch, n_stocks, d_model)
    3. 每只股票输出预测
    
    这样模型可以学习：
    - 消费板块内部联动 (茅台带动五粮液)
    - 金融板块同步波动
    - 跨板块领先滞后关系 (金融领先周期)
    """
    
    def __init__(self,
                 input_dim: int,
                 seq_len: int = 30,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 n_stocks: int = 20,
                 output_dim: int = 1):
        super(iTransformerEncoder, self).__init__()
        
        self.n_stocks = n_stocks
        self.d_model = d_model
        
        # 时间序列编码器 (每只股票共享)
        self.temporal_encoder = TemporalEncoder(seq_len, input_dim, d_model, dropout)
        
        # 股票embedding - 可学习的股票特征
        self.stock_embedding = nn.Parameter(torch.randn(n_stocks, d_model) * 0.02)
        
        # iTransformer层 - 股票间attention
        self.layers = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 收益预测头
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 方向预测头 (辅助任务)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )
        
        # 存储股票间attention权重
        self.stock_attention_weights = None
        
    def forward(self, x: torch.Tensor, return_direction: bool = False):
        """
        Args:
            x: (batch, n_stocks, seq_len, input_dim) - 多股票输入
               或 (batch, seq_len, input_dim) - 单股票输入 (会自动处理)
        Returns:
            return_pred: (batch, n_stocks, output_dim)
            direction_pred: (batch, n_stocks, 1) if return_direction
        """
        # 处理单股票输入的情况
        if x.dim() == 3:
            batch_size = x.size(0)
            # 单股票模式: 复制为多股票格式用于训练
            x = x.unsqueeze(1)  # (batch, 1, seq_len, input_dim)
            single_stock_mode = True
        else:
            batch_size = x.size(0)
            single_stock_mode = False
        
        n_stocks = x.size(1)
        
        # 1. 时间编码 - 每只股票独立编码
        stock_embeds = []
        for i in range(n_stocks):
            stock_seq = x[:, i, :, :]  # (batch, seq_len, input_dim)
            embed = self.temporal_encoder(stock_seq)  # (batch, d_model)
            stock_embeds.append(embed)
        
        # (batch, n_stocks, d_model)
        stock_features = torch.stack(stock_embeds, dim=1)
        
        # 添加可学习的股票embedding
        if n_stocks == self.n_stocks:
            stock_features = stock_features + self.stock_embedding.unsqueeze(0)
        
        # 2. iTransformer层 - 股票间attention
        for layer in self.layers:
            stock_features = layer(stock_features)
            
        # 保存最后一层的attention权重
        self.stock_attention_weights = self.layers[-1].attention.attention_weights
        
        stock_features = self.norm(stock_features)  # (batch, n_stocks, d_model)
        
        # 3. 预测
        return_pred = self.return_head(stock_features)  # (batch, n_stocks, output_dim)
        
        if single_stock_mode:
            return_pred = return_pred.squeeze(1)  # (batch, output_dim)
        
        if return_direction:
            direction_pred = self.direction_head(stock_features)
            if single_stock_mode:
                direction_pred = direction_pred.squeeze(1)
            return return_pred, direction_pred
        
        return return_pred


class TransformerEncoder(nn.Module):
    """
    增强版Transformer编码器模型
    
    改进:
    1. 多尺度卷积预处理 - 捕捉不同时间尺度的模式
    2. 残差连接 - 防止梯度消失
    3. 层归一化 - 稳定训练
    4. 时间注意力池化 - 更好的序列聚合
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 多尺度卷积特征提取
        self.multi_scale_conv = MultiScaleConv(input_dim, d_model)
        
        # 输入嵌入层（带残差连接）
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 时间聚合注意力（学习不同时间步的重要性）
        self.time_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 股票特征增强层
        self.stock_enhance = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 收益率预测头
        self.return_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 方向预测头（辅助任务）
        self.direction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # 输出-1到1，表示方向强度
        )
        
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor, return_direction: bool = False):
        # 多尺度卷积特征
        conv_features = self.multi_scale_conv(x)  # (batch, seq, d_model)
        
        # 线性嵌入（残差）
        embed_features = self.input_embedding(x)  # (batch, seq, d_model)
        
        # 特征融合
        x = conv_features + embed_features
        x = self.embedding_norm(x)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # 时间聚合 - 结合平均池化和注意力池化
        # 平均池化
        avg_pool = x.mean(dim=1)  # (batch, d_model)
        
        # 注意力池化
        attn_weights = self.time_attention(x)  # (batch, seq, 1)
        self.attention_weights = attn_weights.detach()
        attn_pool = (x * attn_weights).sum(dim=1)  # (batch, d_model)
        
        # 拼接两种池化结果
        combined = torch.cat([avg_pool, attn_pool], dim=-1)  # (batch, d_model*2)
        
        # 特征增强
        enhanced = self.stock_enhance(combined)  # (batch, d_model*2)
        
        # 收益率预测
        return_pred = self.return_head(enhanced)  # (batch, output_dim)
        
        if return_direction:
            # 方向预测
            direction_pred = self.direction_head(enhanced)  # (batch, 1)
            return return_pred, direction_pred
        
        return return_pred


class TransformerPredictor:
    """
    Transformer预测器
    
    封装数据处理、模型训练、预测评估的完整流程
    支持两种模式：
    1. 传统Transformer: 时间维度attention
    2. iTransformer (2024): 股票维度attention，学习多股票关联
    """
    
    def __init__(self,
                 seq_length: int = 20,
                 pred_length: int = 5,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = None,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 epochs: int = 150,
                 learning_rate: float = 0.0005,
                 use_itransformer: bool = True):  # 默认使用iTransformer
        """
        参数:
            seq_length: 输入序列长度（回看天数）
            pred_length: 预测长度（预测未来天数）
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: FFN隐层维度，默认4*d_model
            dropout: Dropout率
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            use_itransformer: 是否使用iTransformer多股票关联模式
        """
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_itransformer = use_itransformer
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.stock_codes = None  # 保存股票顺序用于iTransformer
    
    def _prepare_sequences(self, 
                          data: np.ndarray, 
                          target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据
        
        将原始数据转换为模型输入的序列格式
        
        参数:
            data: 特征数据 (n_samples, n_features)
            target: 目标数据 (n_samples,)
        
        返回:
            X: 输入序列 (n_sequences, seq_length, n_features)
            y: 目标值 (n_sequences,)
        """
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(target[i + self.seq_length])
        
        return np.array(X), np.array(y)
    
    def train_and_predict(self,
                         feature_data: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'future_return_5d',
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.15) -> Dict:
        """
        训练模型并进行预测
        
        参数:
            feature_data: 特征数据DataFrame
            feature_cols: 特征列名列表
            target_col: 目标列名
            train_ratio: 训练集比例（默认70%）
            val_ratio: 验证集比例（默认15%，用于早停）
            (测试集 = 1 - train_ratio - val_ratio = 15%)
        
        返回:
            包含预测结果和评估指标的字典
        """
        self.feature_cols = feature_cols
        
        # 固定随机种子
        set_seed(42)
        
        # 按股票分组训练
        all_predictions = []
        all_actuals = []
        all_dates = []
        all_stocks = []
        
        stock_codes = sorted(feature_data['stock_code'].unique())
        self.stock_codes = stock_codes
        n_stocks = len(stock_codes)
        
        model_type = "iTransformer (多股票关联)" if self.use_itransformer else "增强Transformer"
        print(f"\n  模型类型: {model_type}")
        print(f"  模型参数:")
        print(f"    - 股票数量: {n_stocks}")
        print(f"    - 序列长度: {self.seq_length}")
        print(f"    - 预测天数: {self.pred_length}")
        print(f"    - 模型维度: {self.d_model}")
        print(f"    - 注意力头数: {self.n_heads}")
        print(f"    - 编码器层数: {self.n_layers}")

        print(f"    - 训练轮数: {self.epochs}")
        print(f"    - 学习率: {self.learning_rate}")
        print(f"    - 设备: {device}")
        
        # 批量训练所有股票（充分利用GPU）
        print(f"\n  准备全局训练数据...")
        
        if self.use_itransformer:
            # iTransformer多股票同步训练模式
            # 构建 (batch, n_stocks, seq_len, features) 格式数据
            print(f"  [iTransformer模式] 构建多股票同步数据...")
            
            # 1. 首先为每只股票准备原始序列数据（标准化推迟到划分后）
            stock_data_dict = {}
            available_dates = None
            
            for stock_code in stock_codes:
                stock_df = feature_data[feature_data['stock_code'] == stock_code].copy()
                stock_df = stock_df.sort_values('date').reset_index(drop=True)
                
                if len(stock_df) < self.seq_length + 50:
                    continue
                
                available_cols = [c for c in feature_cols if c in stock_df.columns]
                X_raw = stock_df[available_cols].values
                y_raw = stock_df[target_col].values
                dates = stock_df['date'].values
                
                # 先构建序列（用原始数据），标准化推迟到train/test划分后
                X_seq, y_seq = self._prepare_sequences(X_raw, y_raw)
                date_seq = dates[self.seq_length:]
                
                stock_data_dict[stock_code] = {
                    'X': X_seq,  # 未标准化的原始序列
                    'y': y_seq, 
                    'dates': date_seq,
                }
                
                # 取所有股票日期的交集
                if available_dates is None:
                    available_dates = set(date_seq)
                else:
                    available_dates = available_dates.intersection(set(date_seq))
            
            # 2. 只保留所有股票都有数据的日期（同步对齐）
            available_dates = sorted(list(available_dates))
            print(f"    多股票同步日期数: {len(available_dates)}")
            
            # 3. 构建多股票同步数据 (n_dates, n_stocks, seq_len, features)
            active_stocks = list(stock_data_dict.keys())
            n_stocks = len(active_stocks)
            n_dates = len(available_dates)
            seq_len = self.seq_length
            n_features = stock_data_dict[active_stocks[0]]['X'].shape[2]
            
            X_multi = np.zeros((n_dates, n_stocks, seq_len, n_features))
            y_multi = np.zeros((n_dates, n_stocks))
            
            for d_idx, date in enumerate(available_dates):
                for s_idx, stock_code in enumerate(active_stocks):
                    stock_info = stock_data_dict[stock_code]
                    date_idx = np.where(stock_info['dates'] == date)[0][0]
                    X_multi[d_idx, s_idx] = stock_info['X'][date_idx]
                    y_multi[d_idx, s_idx] = stock_info['y'][date_idx]
            
            # 4. 划分训练集、验证集和测试集 (70/15/15)
            split_train = int(n_dates * train_ratio)
            split_val = int(n_dates * (train_ratio + val_ratio))
            
            # 5. 防数据泄露：每只股票独立，仅在训练集上fit StandardScaler
            print(f"    [防泄露] StandardScaler 仅在训练集({split_train}天)上fit...")
            print(f"    数据划分: 训练={split_train}天, 验证={split_val-split_train}天, 测试={n_dates-split_val}天")
            stock_scalers = {}
            for s_idx, stock_code in enumerate(active_stocks):
                scaler = StandardScaler()
                # 将训练期的所有序列展平为(n_train_dates * seq_len, n_features)
                train_flat = X_multi[:split_train, s_idx].reshape(-1, n_features)
                scaler.fit(train_flat)
                stock_scalers[stock_code] = scaler
                # 对训练集、验证集和测试集分别 transform
                for d_idx in range(n_dates):
                    X_multi[d_idx, s_idx] = scaler.transform(X_multi[d_idx, s_idx])
            
            X_train = X_multi[:split_train]
            y_train = y_multi[:split_train]
            X_val = X_multi[split_train:split_val]
            y_val = y_multi[split_train:split_val]
            X_test = X_multi[split_val:]
            y_test = y_multi[split_val:]
            
            train_dates = available_dates[:split_train]
            val_dates = available_dates[split_train:split_val]
            test_dates = available_dates[split_val:]
            
            date_val_all = []
            stock_val_all = []
            for date in val_dates:
                for stock in active_stocks:
                    date_val_all.append(date)
                    stock_val_all.append(stock)
            
            date_test_all = []
            stock_test_all = []
            for date in test_dates:
                for stock in active_stocks:
                    date_test_all.append(date)
                    stock_test_all.append(stock)
            
            # 展平y用于后续评估
            y_val_flat = y_val.flatten()
            y_test_flat = y_test.flatten()
            
            print(f"  多股票训练集: {X_train.shape} (日期数, 股票数, 序列长, 特征数)")
            print(f"  多股票验证集: {X_val.shape}")
            print(f"  多股票测试集: {X_test.shape}")
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)  # (n_dates, n_stocks, 1) 匹配模型输出
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1).to(device)  # (n_dates, n_stocks, 1)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            large_batch_size = min(self.batch_size, len(X_train) // 2)
            
        else:
            # 传统模式：混合所有股票
            X_train_all, y_train_all = [], []
            X_val_all, y_val_all = [], []
            X_test_all, y_test_all = [], []
            date_val_all, stock_val_all = [], []
            date_test_all, stock_test_all = [], []
            
            for stock_code in stock_codes:
                stock_df = feature_data[feature_data['stock_code'] == stock_code].copy()
                stock_df = stock_df.sort_values('date').reset_index(drop=True)
                
                if len(stock_df) < self.seq_length + 50:
                    continue
                
                available_cols = [c for c in feature_cols if c in stock_df.columns]
                X_raw = stock_df[available_cols].values
                y_raw = stock_df[target_col].values
                dates = stock_df['date'].values
                
                # 防数据泄露：先划分再标准化
                X_seq, y_seq = self._prepare_sequences(X_raw, y_raw)
                date_seq = dates[self.seq_length:]
                
                if len(X_seq) < 50:
                    continue
                
                split_train = int(len(X_seq) * train_ratio)
                split_val = int(len(X_seq) * (train_ratio + val_ratio))
                
                # Scaler 仅在训练集上 fit
                scaler = StandardScaler()
                train_flat = X_seq[:split_train].reshape(-1, X_seq.shape[2])
                scaler.fit(train_flat)
                # 分别 transform
                X_train_scaled = np.array([scaler.transform(x) for x in X_seq[:split_train]])
                X_val_scaled = np.array([scaler.transform(x) for x in X_seq[split_train:split_val]])
                X_test_scaled = np.array([scaler.transform(x) for x in X_seq[split_val:]])
                
                X_train_all.append(X_train_scaled)
                y_train_all.append(y_seq[:split_train])
                
                X_val_all.append(X_val_scaled)
                y_val_all.append(y_seq[split_train:split_val])
                date_val_all.extend(date_seq[split_train:split_val])
                stock_val_all.extend([stock_code] * len(y_seq[split_train:split_val]))
                
                X_test_all.append(X_test_scaled)
                y_test_all.append(y_seq[split_val:])
                date_test_all.extend(date_seq[split_val:])
                stock_test_all.extend([stock_code] * len(y_seq[split_val:]))
            
            X_train = np.vstack(X_train_all)
            y_train = np.concatenate(y_train_all)
            X_val = np.vstack(X_val_all)
            y_val = np.concatenate(y_val_all)
            X_test = np.vstack(X_test_all)
            y_test = np.concatenate(y_test_all)
            y_val_flat = y_val
            y_test_flat = y_test
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            large_batch_size = min(self.batch_size, len(X_train) // 4)
        
        print(f"  全局训练集大小: {len(X_train)}")
        print(f"  全局验证集大小: {len(X_val)}")
        print(f"  全局测试集大小: {len(X_test)}")
        
        # 注意: X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor
        # 已在上面的 if/else 分支中根据模式正确创建（含shape处理）
        print(f"  使用大批量训练: Batch Size = {large_batch_size}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # num_workers=0 确保数据加载确定性（可复现）
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=large_batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # 初始化模型 - 根据配置选择iTransformer或传统Transformer
        if self.use_itransformer:
            # iTransformer: (batch, n_stocks, seq_len, features)
            input_dim = X_train.shape[3]
            actual_n_stocks = X_train.shape[1]
        else:
            # 传统Transformer: (batch, seq_len, features)
            input_dim = X_train.shape[2]
            actual_n_stocks = n_stocks
        
        if self.use_itransformer:
            print(f"\n  使用iTransformer架构 (2024 ICLR): 股票间关联建模")
            print(f"    - 学习板块联动: 消费、金融、制造等")
            print(f"    - 捕捉领先滞后关系")
            print(f"    - 股票间Attention矩阵: {actual_n_stocks}x{actual_n_stocks}")
            print(f"    - d_model={self.d_model}, d_ff={self.d_ff}, n_layers={self.n_layers}, n_heads={self.n_heads}")
            self.model = iTransformerEncoder(
                input_dim=input_dim,
                seq_len=self.seq_length,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                n_stocks=actual_n_stocks
            ).to(device)
        else:
            print(f"\n  使用增强Transformer架构")
            self.model = TransformerEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                dropout=self.dropout
            ).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 损失函数和优化器
        # 注意: HuberLoss(delta=0.01)会导致收益回归梯度被方向损失淹没(delta太小)
        # 改用MSELoss让回归损失和方向损失在同一量级
        return_criterion = nn.MSELoss()  # 收益预测损失
        direction_criterion = nn.MSELoss()  # 方向预测损失
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        # 训练模型
        print(f"\n  开始全局模型训练...")
        self.model.train()
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience = 30  # 更严格的早停，防止过拟合
        patience_counter = 0
        accumulation_steps = max(1, 512 // large_batch_size)  # 梯度累积，等效batch_size=512
        print(f"  梯度累积步数: {accumulation_steps} (等效Batch Size = {large_batch_size * accumulation_steps})")
        print(f"  早停: 基于验证集loss (patience={patience})")
        
        for epoch in range(self.epochs):
            # === 训练阶段 ===
            self.model.train()
            epoch_loss = 0
            optimizer.zero_grad()
            for step, (batch_X, batch_y) in enumerate(train_loader):
                # 将数据移至GPU
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # 双任务前向传播
                return_pred, direction_pred = self.model(batch_X, return_direction=True)
                
                # 计算方向标签 (1 for positive, -1 for negative)
                direction_label = torch.sign(batch_y)
                
                # 收益损失 + 方向损失（辅助任务）
                return_loss = return_criterion(return_pred, batch_y)
                direction_loss = direction_criterion(direction_pred, direction_label)
                
                # 总损失 = 收益损失 + 0.1 * 方向损失
                loss = (return_loss + 0.1 * direction_loss) / accumulation_steps
                
                loss.backward()
                epoch_loss += loss.item() * accumulation_steps
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)
            
            # === 验证阶段 ===
            self.model.eval()
            with torch.no_grad():
                val_return_pred, val_direction_pred = self.model(X_val_tensor, return_direction=True)
                val_direction_label = torch.sign(y_val_tensor)
                val_return_loss = return_criterion(val_return_pred, y_val_tensor)
                val_direction_loss = direction_criterion(val_direction_pred, val_direction_label)
                val_loss = (val_return_loss + 0.1 * val_direction_loss).item()
            
            # 早停：基于验证集loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                # 保存最佳模型权重
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    早停于第 {epoch+1} 轮, 最佳验证loss: {best_val_loss:.6f}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 恢复最佳模型权重
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"  已恢复最佳验证loss时的模型权重 (val_loss={best_val_loss:.6f})")
        
        # 保存checkpoint用于复现
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'model_checkpoint.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': input_dim,
                'seq_len': self.seq_length,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
                'n_stocks': actual_n_stocks if self.use_itransformer else n_stocks,
                'use_itransformer': self.use_itransformer,
            },
            'best_val_loss': best_val_loss,
            'stock_scalers': {k: {'mean': v.mean_, 'scale': v.scale_} for k, v in stock_scalers.items()} if self.use_itransformer else {},
        }, ckpt_path)
        print(f"  模型checkpoint已保存: {ckpt_path}")
        
        # 预测
        print(f"\n  生成预测结果...")
        self.model.eval()
        
        # === 验证集预测 ===
        val_predictions_list = []
        val_dataset = TensorDataset(X_val_tensor)
        val_pred_loader = DataLoader(val_dataset, batch_size=large_batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_X, in val_pred_loader:
                pred, _ = self.model(batch_X, return_direction=True)
                val_predictions_list.extend(pred.cpu().numpy().flatten())
        
        val_preds = np.array(val_predictions_list)
        val_acts = np.array(y_val_flat)
        val_min_len = min(len(val_preds), len(val_acts))
        val_preds = val_preds[:val_min_len]
        val_acts = val_acts[:val_min_len]
        
        val_results_df = pd.DataFrame({
            'date': date_val_all[:val_min_len],
            'stock_code': stock_val_all[:val_min_len],
            'actual': val_acts,
            'predicted': val_preds
        })
        
        val_direction_correct = np.sum((val_preds > 0) == (val_acts > 0))
        val_direction_accuracy = val_direction_correct / len(val_preds) if len(val_preds) > 0 else 0
        val_mse = mean_squared_error(val_acts, val_preds)
        
        print(f"  验证集预测: {len(val_preds)}条, 方向准确率={val_direction_accuracy:.4f}, MSE={val_mse:.6f}")
        
        # === 测试集预测 ===
        all_predictions = []
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=large_batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_X, in test_loader:
                pred, _ = self.model(batch_X, return_direction=True)
                # iTransformer: pred shape = (batch, n_stocks, 1) 或 (batch, n_stocks)
                # 传统Transformer: pred shape = (batch, 1)
                all_predictions.extend(pred.cpu().numpy().flatten())
        
        all_actuals = y_test_flat  # 使用展平后的y_test
        all_dates = date_test_all
        all_stocks = stock_test_all
        
        # 计算评估指标
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        
        # 确保长度一致
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        # 方向准确率
        direction_correct = np.sum((predictions > 0) == (actuals > 0))
        direction_accuracy = direction_correct / len(predictions)
        
        # 整理结果
        results_df = pd.DataFrame({
            'date': all_dates[:min_len],
            'stock_code': all_stocks[:min_len],
            'actual': actuals,
            'predicted': predictions
        })
        
        # 保存股票间attention权重用于可视化
        stock_attn = None
        if self.use_itransformer and hasattr(self.model, 'stock_attention_weights'):
            stock_attn = self.model.stock_attention_weights
        
        return {
            'predictions': results_df,          # 测试集预测（最终报告用）
            'val_predictions': val_results_df,   # 验证集预测（Optuna选参用）
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'val_mse': val_mse,
            'val_direction_accuracy': val_direction_accuracy,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_train_loss': avg_train_loss,
            'model': self.model,
            'stock_attention': stock_attn,
            'stock_codes': self.stock_codes
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions


# 测试代码
if __name__ == '__main__':
    # 简单测试
    print("Transformer模型测试")
    print(f"使用设备: {device}")
    
    # 创建模拟数据
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02
    
    # 测试模型
    predictor = TransformerPredictor(
        seq_length=20,
        d_model=32,
        n_heads=2,
        n_layers=1,
        epochs=10
    )
    
    # 创建测试DataFrame
    dates = pd.date_range('2023-01-01', periods=n_samples)
    test_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    test_df['date'] = dates
    test_df['stock_code'] = 'TEST'
    test_df['future_return_5d'] = y
    
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    results = predictor.train_and_predict(
        feature_data=test_df,
        feature_cols=feature_cols,
        train_ratio=0.8
    )
    
    print(f"\nMSE: {results['mse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    print(f"方向准确率: {results['direction_accuracy']:.2%}")
