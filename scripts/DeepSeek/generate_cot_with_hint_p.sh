#!/bin/bash
#SBATCH -J Test-Script
#SBATCH -o logs_and_outputs/Test-Script-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH -w gpu08


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)
# nvidia-smi
# ###################################################################
cd ~/Immunization

model_name="DeepSeek-R1-Distill-Qwen-14B"
n_generation=1
model_path=/home/share/models/
# model_path="/home/dwu/LLaMA-Factory/models"
# model_path="/home/dwu/local_models"
temperature=0.6
# dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
dataset_name="wildjailbreak_train"
dataset_path=""
max_new_tokens=4096
start_idx=0
num_samples=5000
end_idx=$((start_idx + num_samples))
system_prompt=14
moderation_model="wildguard"
# tag="self_align_v2"
tag="self_align_v2"
# tag="self_align_v4"
# system_prompt=12
hint=16

raw_data_file_path="/home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_0-5000_temp_0.6_n_4_sys_0_wildguard_as_judge_labelled.json"
prefix_1="100%"
python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset ${raw_data_file_path} --prefix ${prefix_1} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag $tag --save_name evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}.json

python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}.json --save_name evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}_${moderation_model}.json

prefix_2=0
python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}_${moderation_model}.json --prefix ${prefix_2} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint $hint --save_name evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}_${prefix_2}_with_hint_${hint}.json

python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}_${prefix_2}_with_hint_${hint}.json --save_name evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix_1}_${prefix_2}_with_hint_${hint}_${moderation_model}.json 

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