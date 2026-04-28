#!/bin/bash
#SBATCH -J Test-Script
#SBATCH -o logs_and_outputs/Test-Script-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH -w gpu09


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)
nvidia-smi
# ###################################################################
cd ~/Immunization

model_name="DeepSeek-R1-Distill-Qwen-14B"
n_generation=1
model_path=/home/share/models/
temperature=0.6
dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
max_new_tokens=4096
start_idx=0
num_samples=5000
end_idx=$((start_idx + num_samples))
system_prompt=9 # RealSafe

tag=""
# system_prompt=12

raw_dataset_file="evaluate/results/wildjailbreak_train/Qwen3-14B_sys_0.json"

python evaluate/generate_responses_v4.py --model ${model_path}/${model_name} --dataset ${dataset_name} --max_new_tokens $max_new_tokens --num_samples $num_samples --temperature ${temperature} --n_generation ${n_generation} --need_system_prompt $system_prompt --start_idx ${start_idx}

python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train/${model_name}_${start_idx}-${end_idx}_temp_${temperature}_n_${n_generation}_sys_${system_prompt}.json --moderation wildguard