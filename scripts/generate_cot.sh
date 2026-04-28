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
nvidia-smi
# ###################################################################
cd ~/Immunization

model_name="GLM-Z1-9B-0414"
n_generation=1
# model_path=/home/share/models/
# model_path="/home/dwu/LLaMA-Factory/models"
model_path="/home/dwu/local_models"
temperature=0.6
# dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
dataset_name="wildjailbreak_train"
dataset_path=""
max_new_tokens=4096
start_idx=0
num_samples=5000
end_idx=$((start_idx + num_samples))
system_prompt=0
moderation_model="wildguard"
tag="self_align"
# system_prompt=12

raw_dataset_file="evaluate/results/wildjailbreak_train/Qwen3-14B_sys_0.json"

python evaluate/generate_responses_v4.py --model ${model_path}/${model_name} --dataset evaluate/harmful_questions/${dataset_name}.json --max_new_tokens $max_new_tokens --num_samples $num_samples --temperature ${temperature} --n_generation ${n_generation} --need_system_prompt $system_prompt --start_idx ${start_idx}

python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/${dataset_name}/${model_name}_${start_idx}-${end_idx}_temp_${temperature}_n_${n_generation}_sys_${system_prompt}.json --moderation $moderation_model


# python evaluate/context_distillation_v7.py --model ${model_path}/${model_name} --dataset $raw_dataset_file --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag ${tag}

# python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random.json --moderation wildguard

# python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random_wildguard_as_judge_labelled.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag self_align --hint 1

# # python evaluate/context_distillation_with_hint.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random_wildguard_as_judge_labelled.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag $tag --hint 1 --recheck 1

# python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train/${model_name}_sys_0_temp_${temperature}_n_${n_generation}_${tag}_prefix_random_with_hint.json --moderation wildguard

# python evaluate/context_distillation_v6.py --model ${model_path}/${model_name} --dataset /home/dwu/Immunization/evaluate/harmful_questions/wildjailbreak_bad_case_v2/bad_cases.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag SA --hint 1

# python evaluate/context_distillation_v6.py --model ${model_path}/${model_name} --dataset evaluate/harmful_questions/wildjailbreak_bad_case/bad_case.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag SA --hint 1
# steps_list=("4" "5")
# for step in "${steps_list[@]}"; do
#     python evaluate/context_distillation_v6.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json --prefix $step  --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag SA
# done


# model_name="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_3_STaR_r64_64_5e-5_cosine_bs_4_ep_3"
# n_generation=1
# # model_path=/home/share/models/
# model_path="/home/dwu/LLaMA-Factory/models"
# # model_path="/home/dwu/local_models"
# temperature=0.6
# dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
# max_new_tokens=4096
# start_idx=0
# num_samples=-1
# end_idx=$((start_idx + num_samples))
# system_prompt=11

# python evaluate/context_distillation_v6.py --model ${model_path}/${model_name} --dataset /home/dwu/Immunization/evaluate/harmful_questions/wildjailbreak_bad_case_v2/bad_cases.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag SA --hint 1


# python evaluate/eval_reward.py --response_file evaluate/results/wildjailbreak_train/${model_name}_${start_idx}-${end_idx}_temp_${temperature}_n_${n_generation}_sys_${system_prompt}_wildguard_as_judge_labelled.json --batch_size 16