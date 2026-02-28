#!/bin/bash
# ============================================================
# Optuna 两阶段超参数搜索启动脚本
# ============================================================
# 用法:
#   ./run_optuna.sh                    # 默认: 阶段1=20, 阶段2=16, 100只股票
#   ./run_optuna.sh 20 16              # 阶段1=20次, 阶段2=16次
#   ./run_optuna.sh 20 16 100          # 阶段1=20, 阶段2=16, 100只股票
#   ./run_optuna.sh 0 16 100           # 跳过阶段1，只跑阶段2=16次
#
# 日志: 追加到 run_optuna_output.log（不覆盖历史）
# 数据: 每个trial即时保存到 output/optuna_trial_log.csv
# 数据库: output/optuna_study.db (支持中断恢复)
# ============================================================

set -e

# 参数
PHASE1_TRIALS=${1:-20}
PHASE2_TRIALS=${2:-16}
N_STOCKS=${3:-100}

echo "========================================"
echo " Optuna 两阶段超参数搜索"
echo " 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo " 阶段1（模型搜索）: ${PHASE1_TRIALS} trials"
echo " 阶段2（策略搜索）: ${PHASE2_TRIALS} trials"
echo " 股票数量: ${N_STOCKS}"
echo "========================================"

# conda环境
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate eco_design 2>/dev/null || true
fi

cd "$(dirname "$0")"

# 检查依赖
python -c "import optuna" 2>/dev/null || {
    echo "安装 optuna..."
    pip install optuna -q
}

# PyTorch显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 构建命令
CMD_ARGS="--phase1-trials ${PHASE1_TRIALS} --phase2-trials ${PHASE2_TRIALS} --stocks ${N_STOCKS}"

if [ "$PHASE1_TRIALS" -eq 0 ]; then
    CMD_ARGS="${CMD_ARGS} --skip-phase1"
    echo "跳过阶段1，直接进入阶段2..."
fi

echo "开始搜索..."
python optuna_search.py ${CMD_ARGS}

echo ""
echo "========================================"
echo " 搜索完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo " 查看结果:"
echo "   cat output/optuna_trial_log.csv"
echo "   cat output/optuna_best_params.json"
echo "   cat output/optuna_best_params.txt"
echo "========================================"
