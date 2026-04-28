#!/bin/bash
#SBATCH -J Eval-Model
#SBATCH -o logs_and_outputs/eval/Eval-Model-test-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 16:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -w gpu20
 
source /home/dwu/miniconda3/etc/profile.d/conda.sh
run_env=immune
conda activate $run_env

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

nvidia-smi

stage1="false"  # 控制是否执行 Safety 测试阶段
stage1_1="false"  # 控制是否执行 Over Safety 测试阶段
stage1_2="true"  # 控制是否执行 Jailbreak 测试阶段
stage2="false"  # 控制是否执行 MMLU 测试阶段
gsm8k_stage="false"  # 控制是否执行 GSM8K 测试阶段
math_stage="false"  # 控制是否执行 Math 测试阶段
human_eval_stage="false"  # 控制是否执行 Human_eval 测试阶段
mmlu_stage="false"  #
system_prompt=0
batch_size=16
num_samples_per_task=("10") 
# moderation_model="wildguard"
moderation_model="MD-Judge"
# moderation_model="Llama-Guard"
max_new_tokens=16000
tensor_parallel_size=1
temperature=0
n_generation=1
model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_iter_1_r64_64_5e-5_cosine_bs_4_ep_3")
# model_list=("")

# safety_dataset_name_list=("Salad-base_set_sub_v1" "PKU-SafeRLHF-QA_subset")
safety_dataset_name_list=("adversarialqa" "MaliciousInstruct" "StrongREJECT")
# jailbreak_dataset_name_list=("CodeChameleon_target_llama-2" "DeepInception_target_llama-2")
jailbreak_dataset_name_list=("wildjailbreak_eval_v1")
# jailbreak_dataset_name_list=("CodeChameleon_target_llama-2_tree" "CodeChameleon_target_llama-2_list" "CodeChameleon_target_llama-2_reverse" "CodeChameleon_target_llama-2_half" "Salad-attack_enhanced_set_sub_v1" "ReNeLLM_target_llama-2" "DeepInception_target_llama-2")
# jailbreak_dataset_name_list=("Salad-attack_enhanced_set_sub_v1" "ReNeLLM_target_llama-2" "DeepInception_target_llama-2")
over_safety_dataset_name_list=("OKTest" "XSTest")

for model_name in "${model_list[@]}"; do
   cd ~/Immunization/
   # base_model=/home/share/models/${model_name}
   # base_model=~/local_models/${model_name}
   base_model=~/LLaMA-Factory/models/${model_name}
   
   if [[ "$stage1" == "true" ]]; then
      echo "--------------Test Safety--------------"
      # IFS=','
      dataset_name_string=""
      for item in "${safety_dataset_name_list[@]}"; do
         if [ -z "$dataset_name_string" ]; then
            dataset_name_string="$item"
         else
            dataset_name_string="$dataset_name_string,$item"
         fi
      done
      echo $dataset_name_string
      python evaluate/generate_responses_pack.py --model $base_model --datasets ${dataset_name_string} --need_system_prompt $system_prompt --max_new_tokens $max_new_tokens --tensor_parallel_size $tensor_parallel_size --temperature $temperature

      for dataset_name in "${safety_dataset_name_list[@]}"; do
         
         echo "Now evaluation on: ${dataset_name}"

         python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation $moderation_model
         # conda deactivate
      done
      
   fi
   
   if [[ "$stage1_2" == "true" ]]; then
      echo "--------------Test JailBreak--------------"
      dataset_name_string=""
      for item in "${jailbreak_dataset_name_list[@]}"; do
         if [ -z "$dataset_name_string" ]; then
            dataset_name_string="$item"
         else
            dataset_name_string="$dataset_name_string,$item"
         fi
      done
      echo $dataset_name_string
      # dataset_name_string=$(printf "%s," "${jailbreak_dataset_name_list[@]}" | sed 's/,$//')
      
      python evaluate/generate_responses_pack.py --model $base_model --datasets ${dataset_name_string} --need_system_prompt $system_prompt --max_new_tokens $max_new_tokens --tensor_parallel_size $tensor_parallel_size --temperature $temperature --n_generation $n_generation

      for dataset_name in "${jailbreak_dataset_name_list[@]}"; do
         
         echo "Now evaluation on: ${dataset_name}"

         python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation $moderation_model
         
      done
   fi

   if [[ "$stage1_1" == "true" ]]; then
      echo "--------------Test Over Safety--------------"
      
      for dataset_name in "${over_safety_dataset_name_list[@]}"; do
         
         echo "Now evaluation on: ${dataset_name}"
         python evaluate/generate_responses_v3.py --model $base_model --dataset evaluate/over_safety/${dataset_name}.json --save_path evaluate/results --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json --need_system_prompt $system_prompt --max_new_tokens $max_new_tokens --tensor_parallel_size $tensor_parallel_size --temperature $temperature

         python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation wildguard --over_safety 1
         # conda deactivate
      done
      
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

   if [[ "$gsm8k_stage" == "true" ]]; then
      echo "--------------Test GSM8K--------------"
      cd ../Immunization/evaluate

      # conda activate realignment
      python eval_gsm8k_zero_shot_v2.py --model $base_model --use_cot_prompt --batch_size $batch_size --system_prompt $system_prompt --max_new_tokens $max_new_tokens
      # conda deactivate

   fi

   if [[ "$math_stage" == "true" ]]; then
      echo "--------------Test Math--------------"
      cd ~/math-evaluation-harness
      bash scripts/run_eval.sh direct $base_model $temperature

   fi

   if [[ "$human_eval_stage" == "true" ]]; then
      echo "--------------Test Human Eval--------------"
      cd ~/Immunization
      # num_samples_per_task=("1" "10")
      
      for num_samples in "${num_samples_per_task[@]}"; do
            
         echo "*** WITH SYSTEM PROMPT ***"
         # conda activate realignment
         
         if [[ "$system_prompt" == "1" ]]; then
            output_variable=$(python evaluate/generate_human_eval_v2.py --model $base_model --use_chat_template --use_system_prompt --num_samples_per_task $num_samples)
         else
            output_variable=$(python evaluate/generate_human_eval_v2.py --model $base_model --use_chat_template --num_samples_per_task $num_samples --max_new_tokens $max_new_tokens)
         fi
         echo $output_variable
         file_path=$(echo "$output_variable" | awk -F 'Completed, please check ' '{print $2}')
         # conda deactivate
         conda activate codex
         echo $file_path
         evaluate_functional_correctness $file_path
         conda deactivate
      done
   fi

   if [[ "$mmlu_stage" == "true" ]]; then
      echo "-------------- Test MMMLU_EN --------------"
      cd /home/dwu/Immunization/evaluate/MMLU/src
      system_prompt_sign=("1")
      for use_system_prompt in "${system_prompt_sign[@]}"; do
         # cd ~/Immunization/MMLU/src
         
         python llama3.py --model_name_or_path $base_model --data_dir ../data/MMLU --num_few_shot 0 --use_system_prompt $system_prompt --over_write 1 --language "EN" --max_new_tokens $max_new_tokens
         # fi
      done
   fi

   cd ~/Immunization/
   python evaluate/collect_results.py --response_file ${model_name}_sys_${system_prompt}

done