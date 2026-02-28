#!/bin/bash
# 多次训练选最优
# 跑3次模型训练，保存每次的预测缓存，最后比较选最好的

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eco_design
cd /root/eco_design/transformer_quant_strategy

N_RUNS=3
BEST_SCORE=-999
BEST_RUN=0
SEEDS=(42 123 2024)

for i in $(seq 1 $N_RUNS); do
    SEED=${SEEDS[$((i-1))]}
    echo ""
    echo "============================================"
    echo "  训练轮次 $i / $N_RUNS  (seed=$SEED)"
    echo "============================================"
    
    # 删除旧缓存，强制重新训练
    rm -f output/cached_model_results.pkl
    
    # 训练模型（跳过Phase1搜索，Phase2=0只训练）
    python -u optuna_search.py --skip-phase1 --phase2-trials 0 2>&1 | tee "output/train_run_${i}.log"
    
    # 保存本轮缓存
    if [ -f "output/cached_model_results.pkl" ]; then
        cp output/cached_model_results.pkl "output/cached_model_results_run${i}.pkl"
        
        # 提取测试集年化收益和夏普比率
        ANNUAL=$(grep "年化收益率:" "output/train_run_${i}.log" | tail -1 | grep -oP '[\d.]+(?=%)')
        SHARPE=$(grep "夏普比率:" "output/train_run_${i}.log" | tail -1 | grep -oP '[\d.-]+' | tail -1)
        DIR_ACC=$(grep "方向准确率:" "output/train_run_${i}.log" | tail -1 | grep -oP '[\d.]+' | tail -1)
        
        echo ""
        echo "  [轮次 $i] 年化=${ANNUAL}%, 夏普=${SHARPE}, 方向准确率=${DIR_ACC}"
        
        # 比较选最优（用夏普比率）
        BETTER=$(python -c "print(1 if float('${SHARPE}') > float('${BEST_SCORE}') else 0)")
        if [ "$BETTER" = "1" ]; then
            BEST_SCORE=$SHARPE
            BEST_RUN=$i
            echo "  >>> 新的最优！ <<<"
        fi
    else
        echo "  [轮次 $i] 训练失败，无缓存文件"
    fi
done

echo ""
echo "============================================"
echo "  最终结果: 最优轮次 = Run ${BEST_RUN} (Sharpe=${BEST_SCORE})"
echo "============================================"

# 将最优轮次的缓存设为默认
cp "output/cached_model_results_run${BEST_RUN}.pkl" "output/cached_model_results.pkl"
echo "  已将 Run ${BEST_RUN} 的预测缓存设为默认"
echo "  现在可以运行 bash run.sh 查看完整结果"
