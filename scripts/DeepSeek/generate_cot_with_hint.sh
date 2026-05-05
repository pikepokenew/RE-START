ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/qwen3_env/bin/"

model_name="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3"
n_generation=1
# model_path=/home/share/models/
model_path="/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
# model_path="/home/dwu/local_models"
temperature=0.6
# dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
dataset_name="wildjailbreak_train"
dataset_path="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/harmful_questions/wildjailbreak_train.json"
max_new_tokens=4096
start_idx=0
num_samples=5000
end_idx=$((start_idx + num_samples))
moderation_model="wildguard"
# tag="self_align_v2"
# tag="self_align_v2_1"
hint=0

${ENV_BIN}python /apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/context_distillation_with_hint.py \
    --model ${model_path} \
    --dataset ${dataset_path} \
    --tag None \
    --hint 0 \
    --prefix 0 \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --tensor_parallel_size 1 \
    --temperature 0.6 \
    --top_p 1.0 \
    --max_new_tokens $max_new_tokens

# 评估
python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random.json --moderation wildguard

# Step 3. HINT生成
python evaluate/context_distillation_with_hint.py --model /apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard.json --tag self_align_v2 --temperature 0.6 --max_new_tokens 4096 --hint 16 --apply_hint 1 --output_path evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard_prefix_0_hint.json

# Step 4. 评估
python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard_prefix_0_hint.json --moderation wildguard

# raw_data_file_path="/home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_0-5000_temp_0.6_n_4_sys_0_wildguard_as_judge_labelled.json"
# prefix="0"
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset ${raw_data_file_path} --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag $tag

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random.json

# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random_wildguard.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint $hint

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}_with_hint_${hint}.json

# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard_as_judge_labelled.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation 1 --temperature $temperature --tag $tag --hint $hint

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_1_${tag}_prefix_${prefix}_with_hint_${hint}.json

# # 消融实验
# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset /home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint 0

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}.json


# 消融实验 取消rnd_mask
# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset ${raw_data_file_path} --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint 0

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}.json

# # # 消融实验 取消general rules
# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint 0

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}.json

# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_0.6_n_1_${tag}_prefix_${prefix}_wildguard.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint 16

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}_with_hint_16.json