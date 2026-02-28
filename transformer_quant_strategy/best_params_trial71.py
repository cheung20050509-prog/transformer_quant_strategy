# -*- coding: utf-8 -*-
"""
Optuna 两阶段搜索最优超参数 - Trial 71
========================================
来源: Phase2 策略搜索 (100 trials, 2026-02-27)
模型: Phase1 最优 (d_model=512, Optuna Trial 8)
综合得分: 0.7098

回测结果:
  - 年化收益率: 40.68%
  - 最大回撤:   5.43%
  - 夏普比率:   1.197
  - 胜率:       42.47%
  - 交易次数:   365
  - 收益/回撤比: 7.49

数据范围: 100只A股, 2018-01-01 ~ 2025-12-31
训练/测试划分: 80/20
"""

# ===== 模型超参数 =====
MODEL_PARAMS = {
    'd_model': 512,
    'd_ff': 2048,          # d_ff_ratio = 4
    'n_heads': 8,
    'n_layers': 16,
    'seq_length': 60,
    'dropout': 0.21538,
    'learning_rate': 3.046e-04,
    'batch_size': 512,
    'epochs': 800,
}

# ===== 策略超参数 =====
STRATEGY_PARAMS = {
    'buy_threshold': 0.01534,      # 预测收益 > 1.534% 时买入
    'sell_threshold': -0.00798,    # 预测收益 < -0.798% 时卖出
    'stop_loss_ratio': 0.06238,    # 止损线 6.24%
    'take_profit_ratio': 0.14166,  # 止盈线 14.17%
    'max_position_ratio': 0.37682, # 最大仓位 37.68%
}

# ===== 回测绩效 =====
PERFORMANCE = {
    'annual_return': 0.4068,
    'max_drawdown': 0.0543,
    'sharpe_ratio': 1.197,
    'win_rate': 0.4247,
    'total_trades': 365,
    'return_drawdown_ratio': 7.49,
}

# ===== 旧 Trial 2 参数（对比用）=====
OLD_TRIAL2_MODEL_PARAMS = {
    'd_model': 1536,
    'd_ff': 6144,          # d_ff_ratio = 4
    'n_heads': 8,
    'n_layers': 16,
    'seq_length': 30,
    'dropout': 0.22106,
    'learning_rate': 5.595e-05,
    'batch_size': 512,
    'epochs': 390,
}

OLD_TRIAL2_STRATEGY_PARAMS = {
    'buy_threshold': 0.01184,
    'sell_threshold': -0.01016,
    'stop_loss_ratio': 0.03109,
    'take_profit_ratio': 0.14665,
    'max_position_ratio': 0.33254,
}

OLD_TRIAL2_PERFORMANCE = {
    'annual_return': 0.4207,
    'max_drawdown': 0.1007,
    'sharpe_ratio': 1.077,
}
