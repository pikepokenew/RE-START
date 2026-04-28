#!/bin/bash
#SBATCH -J Eval_Safety
#SBATCH -o logs_and_outputs/Eval_Safety_test-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 24:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:geforce_rtx_2080_ti:1
#SBATCH -w gpu02

source /home/dwu/miniconda3/etc/profile.d/conda.sh
run_env=resta
conda activate $run_env

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:10233 && export https_proxy=http://127.0.0.1:10233 && export all_proxy=http://127.0.0.1:10233

stage1="true"  # 控制是否执行 Safety 测试阶段
stage1_1="false"  # 控制是否执行 Over Safety 测试阶段
system_prompt=1
batch_size=32
pool_size=100
# model_list=("llama_peft_CodeAlpaca-20k_add_0.1_1k_safety_model" "llama_peft_CodeAlpaca-20k_add_0.2_1k_safety_model" "llama_peft_CodeAlpaca-20k_add_0.3_1k_safety_model" "llama_peft_CodeAlpaca-20k_add_0.4_1k_safety_model" "llama_peft_CodeAlpaca-20k_add_0.5_1k_safety_model")

# model_list=("llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_1024" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_256" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_128" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_64" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_32" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_16" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_8" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_4"
# "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_2" "llama_peft_alpaca_en_TIES_safe_ewc_top_p_100_q_proj_v_proj_blocksize_1")

model_list=("llama_fft_mix_alpaca_zh_alpaca_en_add_0.5_1k_safety_model_linear_v1" "llama_fft_mix_alpaca_hindi_and_alpaca_en_add_0.5_1k_safety_model_linear_v1")


for model_name in "${model_list[@]}"; do
   cd ~/ReAlign/
   base_model=/home/dwu/resta/saved_models/${model_name}
   # base_model=/home/dwu/ModelTailor/saved_models/${model_name}
   dataset_name_list=("catqa_english" "catqa_chinese" "catqa_vietnamese" "harmfulqa" "dangerousqa" "adversarialqa")

   if [[ "$stage1" == "true" ]]; then
      echo "--------------Test Safety--------------"
      if [ -n "$lora_name" ]; then
         
         # lora_name=$(echo "$lora_path" | awk -F'/' '{print $(NF-1)}')
         for dataset_name in "${dataset_name_list[@]}"; do
            echo "model with Lora"
            echo "Now evaluation on: ${dataset_name}"
            # python evaluate/generate_responses.py --model $base_model --lora_path $lora_path --dataset evaluate/harmful_questions/${dataset_name}.json --save_path evaluate/results --batch_size $batch_size --need_system_prompt $system_prompt --save_name ${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json

            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size 16

            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size $batch_size  --moderation "Llama-Guard-ft"
            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size 16 --moderation MD-Judge
            python evaluate/gpt4_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results --pool_size $pool_size
         done

      else
         for dataset_name in "${dataset_name_list[@]}"; do
            
            echo "Now evaluation on: ${dataset_name}"
            # python evaluate/generate_responses.py --model $base_model --dataset evaluate/harmful_questions/${dataset_name}.json --save_path evaluate/results --batch_size $batch_size --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json --need_system_prompt $system_prompt

            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}_${model_name}.json --save_path evaluate/results
            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size 16
            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size $batch_size --moderation "Llama-Guard-ft"
            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size 16 --moderation MD-Judge
            python evaluate/gpt4_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --pool_size $pool_size
         done
      fi
   fi

   if [[ "$stage1_1" == "true" ]]; then
   echo "--------------Test Over Safety--------------"

      dataset_name_list=("OKTest" "XSTest")
      if [ -n "$lora_name" ]; then
         
         # lora_name=$(echo "$lora_path" | awk -F'/' '{print $(NF-1)}')
         for dataset_name in "${dataset_name_list[@]}"; do
            echo "model with Lora"
            echo "Now evaluation on: ${dataset_name}"
            # python evaluate/generate_responses.py --model $base_model --lora_path $lora_path --dataset evaluate/over_safety/${dataset_name}.json --save_path evaluate/results --batch_size $batch_size --need_system_prompt $system_prompt --save_name ${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json

            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results --batch_size 16 --moderation "Over-Refusal"
            python evaluate/gpt4_as_judge.py --response_file evaluate/results/over_safety/${dataset_name}/${model_name}_${lora_name}_sys_${system_prompt}.json --save_path evaluate/results
         done

      else
         for dataset_name in "${dataset_name_list[@]}"; do
            
            echo "Now evaluation on: ${dataset_name}"
            # python evaluate/generate_responses.py --model $base_model --dataset evaluate/over_safety/${dataset_name}.json --save_path evaluate/results --batch_size $batch_size --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json --need_system_prompt $system_prompt

            # python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation "Over-Refusal" --batch_size 16
            python evaluate/gpt4_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --pool_size $pool_size
         done
      fi
   fi

   if [[ "$stage2" == "true" ]]; then
      echo "--------------Test MMLU--------------"

      cd ../mmlu
      if [ -n "$lora_name" ]; then
         python evaluate_hf.py -m $base_model --lora $lora_path
      else
         python evaluate_hf.py -m $base_model
      fi
   fi

   cd ~/ReAlign/
   python evaluate/collect_results.py --response_file ${model_name}_sys_${system_prompt} --judge_model "gpt-4o-ca"

done