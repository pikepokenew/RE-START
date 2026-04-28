#!/bin/bash
#SBATCH -J Test-Script
#SBATCH -o logs_and_outputs/Test-Script-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -w gpu18


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)
# nvidia-smi
# ###################################################################
cd ~/Immunization

# model_name="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v3_rnd_mask_hint5_sft_r64_64_5e-5_cosine_bs_4_ep_3"
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
tag="self_align_v2"
# tag="self_align_v3"
# tag="self_align_v4"
# system_prompt=12
hint=16

raw_data_file_path="/home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_0-5000_temp_0.6_n_4_sys_0_wildguard_as_judge_labelled.json"

# raw_data_file_path="/home/dwu/Immunization/evaluate/results/wildjailbreak_train/Qwen3-14B_temp_0.6_n_1_sys_0.json"

python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset ${raw_data_file_path} --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag $tag

python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random.json

prefix=0
python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random_wildguard_as_judge_labelled.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation ${n_generation} --temperature $temperature --tag $tag --hint $hint

python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_${prefix}_with_hint_${hint}.json

# prefix=0
# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_wildguard_as_judge_labelled.json --prefix ${prefix} --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation 1 --temperature $temperature --tag $tag --hint $hint

# python evaluate/moderation_as_judge_v4.py --moderation wildguard --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_1_${tag}_prefix_${prefix}_with_hint_${hint}.json