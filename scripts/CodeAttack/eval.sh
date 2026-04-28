#!/bin/bash
#SBATCH -J Eval-CodeAttack
#SBATCH -o logs_and_outputs/eval/Eval-CodeAttack-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 24:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -w gpu20

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

nvidia-smi
# ####################### CodeAttack ##################################
# model_path="/home/share/models"
# model_path="/home/dwu/local_models"
model_path="/home/dwu/LLaMA-Factory/models"

max_new_tokens=16000
temperature=0.0
system_prompt=0
moderation_model=MD-Judge
# moderation_model=wildguard
exp_name_list=("python_string_full" "python_list_full" "python_stack_full")
# exp_name_list=("python_list_full")

# model_list=("STAIR-Llama-3.1-8B-DPO-3")
# model_list=("Qwen3-14B_peft_UnSafeChain-selected_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_UnSafeChain-selected_r64_64_5e-5_cosine_bs_4_ep_3")
model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_OpenMathInstruct-2_10k_STAR-S_filtered_Hard_r64_64_5e-5_cosine_bs_4_ep_3")

for model_name in "${model_list[@]}"; do

    for exp_name in "${exp_name_list[@]}"; do
        model_name_or_path=${model_path}/${model_name}
        cd ~/CodeAttack
        python main_test.py --num_sample 1 --prompt-type code_python --judge-model gpt-4o --exp_name $exp_name --data_key code_wrapped_plain_attack --target-max-n-tokens $max_new_tokens --judge --temperature $temperature --start_idx 0 --end_idx -1 --target-model ${model_name_or_path} --system_prompt_type $system_prompt

        cd ~/Immunization
        python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/CodeAttack_${exp_name}_code_wrapped_plain_attack/${model_name}_sys_${system_prompt}.json --moderation $moderation_model
        # moderation
        
    done

done