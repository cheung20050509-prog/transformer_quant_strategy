#!/bin/bash
# Transformer量化交易策略运行脚本
# 使用方法: ./run.sh [--optimize]

# 设置环境
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eco_design

# 进入项目目录
cd /root/eco_design/transformer_quant_strategy

echo "========================================"
echo "Transformer Quantitative Trading Strategy"
echo "========================================"
echo ""

# 运行主程序
if [ "$1" == "--optimize" ]; then
    echo "[Mode] Hyperparameter Optimization with Optuna"
    python -u main.py --optimize --trials ${2:-30}
else
    echo "[Mode] Full Experiment Run"
    python -u main.py
fi

echo ""
echo "========================================"
echo "Done! Results saved to output/"
echo "========================================"
