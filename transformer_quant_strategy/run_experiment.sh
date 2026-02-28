#!/bin/bash
# 量化交易策略实验运行脚本
# 使用方法: ./run_experiment.sh

echo "=========================================="
echo "Transformer量化交易策略实验"
echo "=========================================="
echo ""

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eco_design

# 设置Python环境变量，确保实时输出
export PYTHONUNBUFFERED=1

# 运行主程序
cd /root/eco_design/transformer_quant_strategy
python -u main.py

echo ""
echo "=========================================="
echo "实验完成！结果保存在 output/ 目录"
echo "=========================================="
