#!/bin/bash

# ==========================================
# 基础配置
# ==========================================
ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/qwen3_env/bin/"
model_path="/apdcephfs_nj3/share_300616873/hunyuan/common/Qwen3-14B"
dataset_path="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/harmful_questions/wildjailbreak_train_Qwen3_14B.json"

max_new_tokens=4096
temperature=0.6
# tag="RealSafe-R1"
tag="self_align_v2"
prefix=random

# 任务切分配置
TOTAL_SAMPLES=5000
NUM_GPUS=8

# 计算每张卡处理的样本数 (向上取整)
CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_GPUS - 1) / NUM_GPUS ))

echo "Starting distributed inference on ${NUM_GPUS} GPUs."
echo "Total samples: ${TOTAL_SAMPLES}, Chunk size per GPU: ${CHUNK_SIZE}"

# ==========================================
# 多卡并行派发任务
# ==========================================
for (( i=0; i<NUM_GPUS; i++ ))
do
    # 计算当前分片的 start 和 end
    START_IDX=$(( i * CHUNK_SIZE ))
    END_IDX=$(( (i + 1) * CHUNK_SIZE ))
    
    # 最后一个分片确保不超过 TOTAL_SAMPLES
    if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
        END_IDX=$TOTAL_SAMPLES
    fi
    
    # 如果起始索引已经超出总数，直接退出循环
    if [ $START_IDX -ge $TOTAL_SAMPLES ]; then
        break
    fi

    echo "Launching Job on GPU ${i} | Samples: ${START_IDX} to ${END_IDX}"

    # 指定显卡并在后台运行任务 (&)
    CUDA_VISIBLE_DEVICES=$i ${ENV_BIN}python /apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/context_distillation_with_hint.py \
        --model ${model_path} \
        --dataset ${dataset_path} \
        --tag ${tag} \
        --hint 0 \
        --prefix ${prefix} \
        --start_idx ${START_IDX} \
        --end_idx ${END_IDX} \
        --tensor_parallel_size 1 \
        --temperature ${temperature} \
        --top_p 1.0 \
        --max_new_tokens ${max_new_tokens} &
done

# 等待所有后台任务完成
echo "All jobs dispatched. Waiting for completion..."
wait
echo "All processes finished successfully!"
