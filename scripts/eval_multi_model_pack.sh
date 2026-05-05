# 帮我整理一下这段代码.
nvidia-smi

stage1="true"  # 控制是否执行 Safety 测试阶段
stage1_1="true"  # 控制是否执行 Over Safety 测试阶段
stage1_2="true"  # 控制是否执行 Jailbreak 测试阶段
stage2="false"  # 控制是否执行 MMLU 测试阶段
gsm8k_stage="false"  # 控制是否执行 GSM8K 测试阶段
math_stage="false"  # 控制是否执行 Math 测试阶段
human_eval_stage="false"  # 控制是否执行 Human_eval 测试阶段
mmlu_stage="false"  #
gpqa_stage="false"
arc_stage="false"
system_prompt=0
batch_size=16
num_samples_per_task=("5") 
# moderation_model="wildguard"
moderation_model="MD-Judge"
# moderation_model="Llama-Guard"
max_new_tokens=16000
tensor_parallel_size=1
temperature=0
n_generation=1
tag=""
seed=42
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_fa2_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_RealSafe_sft_5k_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3")
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_glm-4.5_self_align_v2_sft_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_DS-V3.2-Exp-thinking_self_align_v2_sft_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_v2_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_DS-V3.2-Exp-thinking_self_align_v2_sft_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_glm-4.5_self_align_v2_sft_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_v2_r64_64_5e-5_constant_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_ultrafeedback_kimi-k2-thinking_self_align_v2_sft_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3")
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_UnSafeChain-full_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_UnSafeChain-full_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_UnSafeChain-selected_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_UnSafeChain-selected_r64_64_5e-5_cosine_bs_4_ep_3")
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF_r64_64_5e-5_cosine_bs_4_ep_3")
model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_OpenMathInstruct-2_10k_STAR-S_filtered_Hard_r64_64_5e-5_cosine_bs_4_ep_3")

# safety_dataset_name_list=("Salad-base_set_sub_v1" "PKU-SafeRLHF-QA_subset")
safety_dataset_name_list=("advbench" "HEx-PHI")
# jailbreak_dataset_name_list=("CodeChameleon_target_llama-2" "DeepInception_target_llama-2")
# jailbreak_dataset_name_list=("wildjailbreak_eval_v1")
# jailbreak_dataset_name_list=("Chipher_SelfDefine")
# jailbreak_dataset_name_list=("Chipher_SelfDefine" "Chipher_Morse" "Chipher_Ascii" "Chipher_Caesar" )
jailbreak_dataset_name_list=("CodeChameleon_target_llama-2_tree" "CodeChameleon_target_llama-2_list" "CodeChameleon_target_llama-2_reverse" "CodeChameleon_target_llama-2_half" "Salad-attack_enhanced_set_sub_v1" "ReNeLLM_target_llama-2" "DeepInception_target_llama-2" "wildjailbreak_eval_v1")
# jailbreak_dataset_name_list=("wildjailbreak_eval_v1")
over_safety_dataset_name_list=("OKTest" "XSTest")
# over_safety_dataset_name_list=("or-bench_1k" "or-bench-hard-1k")

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
      # bash scripts/run_eval.sh direct $base_model 0.6 "${tag}" ${seed}

      OUTPUT_DIR=results/math_eval
      # DATA_NAMES="gsm8k,minerva_math,gsm_hard"
      # DATA_NAMES="gsm8k,minerva_math,gsm_hard"
      # DATA_NAMES="gsm8k,minerva_math,gsm_hard"
      # DATA_NAMES="math,math_oai,minerva_math,gsm8k,gsm_hard"
      # DATA_NAMES="gsm8k,math_oai,minerva_math,math,AQuA,mmlu_stem,sat_math"
      # DATA_NAMES="gsm8k,math_oai,AQuA,mmlu_stem,sat_math"
      DATA_NAMES="gsm8k,minerva_math"
      # DATA_NAMES="minerva_math,simuleq,numglue"
      # DATA_NAMES="gsm8k,minerva_math,AQuA,simuleq,numglue,mmlu_stem,sat_math"
      # DATA_NAMES="AQuA"
      # DATA_NAMES="gsm8k,minerva_math,svamp,asdiv,mawps,tabmwp,mathqa,mmlu_stem,sat_math"
      SPLIT="test"
      NUM_TEST_SAMPLE=-1
      PROMPT_TYPE="direct"

      # single-gpu
      CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
      python3 -u math_eval.py \
         --model_name_or_path ${base_model} \
         --output_dir ${OUTPUT_DIR} \
         --data_names ${DATA_NAMES} \
         --split ${SPLIT} \
         --prompt_type ${PROMPT_TYPE} \
         --num_test_sample ${NUM_TEST_SAMPLE} \
         --seed ${seed} \
         --temperature 0.6 \
         --n_sampling 1 \
         --top_p 1 \
         --start 0 \
         --end -1 \
         --use_vllm \
         --save_outputs \
         --overwrite \
         --system_prompt ${system_prompt} \
         --tag "${tag}" \
         --max_tokens_per_call 16000 \

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
      system_prompt_sign=("0")
      for use_system_prompt in "${system_prompt_sign[@]}"; do
         # cd ~/Immunization/MMLU/src
         
         python llama3.py --model_name_or_path $base_model --data_dir ../data/mmlu_redux --num_few_shot 0 --use_system_prompt $system_prompt --over_write 1 --language "EN" --max_new_tokens $max_new_tokens --tag "${tag}" --seed ${seed}
         # fi
      done
   fi
   if [[ "$gpqa_stage" == "true" ]]; then
      echo "--------------Test GPQA--------------"
      cd ~/Immunization/ 
      python evaluate/eval_gpqa.py --model $base_model  --subset main --max_new_tokens $max_new_tokens --temperature 0.6 --tag "${tag}" --seed ${seed}

   fi
   if [[ "$arc_stage" == "true" ]]; then
      echo "--------------Test ARC--------------"
      cd ~/Immunization/ 
      python evaluate/eval_ai2_arc.py --model $base_model --max_new_tokens $max_new_tokens --subset Challenge --temperature 0.6 --need_system_prompt ${system_prompt}

   fi
   cd ~/Immunization/
   python evaluate/collect_results.py --response_file ${model_name}_sys_${system_prompt}

done