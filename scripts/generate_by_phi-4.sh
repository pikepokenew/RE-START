#!/bin/bash
#SBATCH -J Test-Script
#SBATCH -o logs_and_outputs/Test-Script-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -w gpu20


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)
nvidia-smi
# ###################################################################
cd ~/Immunization

model_name="Phi-4-reasoning"
n_generation=1
# model_path=/home/share/models/
# model_path="/home/dwu/LLaMA-Factory/models"
model_path="/home/dwu/local_models"
temperature=0.8
dataset_name="evaluate/harmful_questions/wildjailbreak_train.json"
max_new_tokens=4096
start_idx=0
num_samples=5000
end_idx=$((start_idx + num_samples))
system_prompt=11
top_k=50
top_p=0.95
# system_prompt=12

# python evaluate/context_distillation_v6.py --model ${model_path}/${model_name} --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json --prefix random --max_new_tokens $max_new_tokens --num_samples $num_samples --n_generation $n_generation --temperature $temperature --tag SA

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

python evaluate/generate_responses_v4.py --model ${model_path}/${model_name} --dataset ${dataset_name} --max_new_tokens $max_new_tokens --num_samples $num_samples --temperature ${temperature} --n_generation ${n_generation} --need_system_prompt $system_prompt --top_k $top_k --top_p $top_p --start_idx ${start_idx}

python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/${model_name}_${start_idx}-${end_idx}_temp_${temperature}_n_${n_generation}_sys_${system_prompt}.json --moderation wildguard

# python evaluate/eval_reward.py --response_file evaluate/results/wildjailbreak_train/${model_name}_${start_idx}-${end_idx}_temp_${temperature}_n_${n_generation}_sys_${system_prompt}_wildguard_as_judge_labelled.json --batch_size 16

# # model_name_or_path="/home/share/models/DeepSeek-R1-Distill-Qwen-14B"
# model_name_or_path="/home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_Qwen3-14B_DA_sft_5k_with_prompt_r64_64_5e-5_cosine_bs_8_ep_3"
# # model_name_or_path="/home/dwu/local_models/Llama-3.1-Nemotron-Nano-8B-v1"
# system_prompt=8
# n_generation=64
# num_samples=100
# temperature=1.0
# max_new_tokens=4096
# # dataset_name="wildjailbreak_train.json"
# dataset_name="wildjailbreak_eval_v1.json"
# # dataset_name="Salad-attack_enhanced_set.json"
# python evaluate/generate_responses_v4.py --model ${model_name_or_path} --dataset evaluate/harmful_questions/${dataset_name} --n_generation $n_generation --max_new_tokens $max_new_tokens --temperature $temperature --need_system_prompt $system_prompt --num_samples $num_samples

# python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_R1_Llama_DA_sft_random_r64_64_5e-5_constant_bs_4_ep_3_adapter_temp_1.0_n_8_sys_8.json --moderation wildguard

# deepspeed --num_gpus=1 test/gcg.py --model_name_or_path ${model_name_or_path}  --save_name test/Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_wildjailbreak_train_temp_1.0_n_8_sys_8.pt --num_samples $num_samples --dataset evaluate/results/wildjailbreak_train/Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_temp_1.0_n_8_sys_8_wildguard_as_judge_labelled.json --batch_size 4

# python evaluate/generate_responses_v4.py --model /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_iter_2_gcg --dataset evaluate/harmful_questions/wildjailbreak_train.json --n_generation 8 --max_new_tokens 4096 --temperature 1.0 --need_system_prompt 0 --num_samples 5000

# python evaluate/moderation_as_judge_v4.py --response_file /home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_iter_2_gcg_temp_1.0_n_8_sys_0.json --moderation wildguard


# python evaluate/generate_responses_v4.py --model /home/share/models/DeepSeek-R1-Distill-Llama-8B --dataset evaluate/harmful_questions/wildjailbreak_train.json --n_generation 8 --max_new_tokens 4096 --temperature 1.0 --need_system_prompt 8 --num_samples 5000

# cd ~/Immunization/test
# python evaluate/moderation_as_judge_v4.py --response_file /home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_temp_1.0_n_8_sys_0.json --moderation wildguard

# deepspeed --num_gpus=1 test/gcg.py --model_name_or_path ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3  --save_name test/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_wildjailbreak_train_temp_1.0_n_8_sys_0.pt --num_samples -1 --dataset /home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_gcg_r64_64_5e-5_constant_bs_4_ep_3_temp_1.0_n_8_sys_0_wildguard_as_judge_labelled.json --batch_size 4

# python evaluate/moderation_as_judge_v4.py --response_file    /home/dwu/Immunization/evaluate/results/wildjailbreak_train/Qwen3-14B_temp_1.0_n_4_sys_8.json --moderation wildguard

# python evaluate/generate_responses_v4.py --model /home/share/models/Qwen3-14B --dataset evaluate/harmful_questions/CodeChameleon_target_llama-2_tree.json --n_generation 10 --max_new_tokens 4096 --temperature 0.6 --need_system_prompt 8

# python evaluate/moderation_as_judge_v4.py --response_file    /home/dwu/Immunization/evaluate/results/CodeChameleon_target_llama-2_tree/CodeChameleon_target_llama-2_tree_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v4_r64_64_5e-5_constant_bs_4_ep_3_sys_0_None_temp_0.6_n_10_prefix_0_v2.json --moderation wildguard

# python evaluate/moderation_as_judge_v4.py --response_file    /home/dwu/Immunization/evaluate/results/CodeChameleon_target_llama-2_tree/CodeChameleon_target_llama-2_tree_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v4_r64_64_5e-5_constant_bs_4_ep_3_sys_0_None_temp_0.6_n_10_prefix_0_v2.json --moderation MD-Judge --num_samples -1
# deepspeed --num_gpus=1 compute_token_entropy_ds.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --num_samples -1 --dataset /home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_DeepSeek-R1-Distill-Qwen-14B_sys_0_deliberative_alignment_v2_temp_1.0_n_3_prefix_1_v2_wildguard_as_judge_labelled.json --save_name wildjailbreak_train_token_entropy.pt --batch_size 4

# deepspeed --num_gpus=1 compute_token_entropy_ds.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --num_samples 100 --dataset /home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_v2_1_info.json --save_name wildjailbreak_train_R1_prefix_max_v2_1_token_entropy.pt --batch_size 4

# 生成数据
# Qwen-8B/14B
# dataset_name="wildjailbreak_train_part2"
# python evaluate/generate_responses_v3.py --model /home/share/models//DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/harmful_questions/${dataset_name}.json --max_new_tokens 4096 --num_samples 10000 --temperature 0.0

# # 用DS-Distill-Qwen-14B生成续写数据
# python evaluate/generate_responses_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --prefix 1 --max_new_tokens 4096 --dataset evaluate/results/${dataset_name}/Qwen3-8B_sys_0.json --tag deliberative_alignment_v2

# python evaluate/generate_responses_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --prefix 1 --max_new_tokens 4096 --dataset evaluate/results/${dataset_name}/Qwen3-14B_sys_0.json --tag deliberative_alignment_v2

# 批量生成的续写数据，其中从第2步-第10步前缀开始续写
# steps_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# for step in "${steps_list[@]}"; do
#     python evaluate/generate_response_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B   --num_samples -1 --dataset evaluate/results/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_sys_0.json  --prefix ${step} --max_new_tokens 4096 --tag deliberative_alignment_v2

#     python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}//DeepSeek-R1-Distill-Qwen-14B_sys_0_sys_0_deliberative_alignment_v2_prefix_${step}.json --moderation wildguard

# done


# dataset_list=("CodeAttack_python_string_full_code_wrapped_plain_attack" "CodeAttack_python_list_full_code_wrapped_plain_attack" "CodeAttack_python_stack_full_code_wrapped_plain_attack" "CodeChameleon_target_llama-2_tree" "CodeChameleon_target_llama-2_list" "CodeChameleon_target_llama-2_reverse" "CodeChameleon_target_llama-2_half" "Salad-attack_enhanced_set_sub_v1" "ReNeLLM_target_llama-2" "DeepInception_target_llama-2" "OKTest" "XSTest")
# tag="only_okay_suffix"
# # dataset_list=("OKTest" "XSTest")
# for dataset_name in "${dataset_list[@]}"; do
#     python evaluate/generate_responses_with_prefix.py --model ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3 --prefix 0 --max_new_tokens 16000 --dataset evaluate/results/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3_sys_0.json --tag ${tag}

#     python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3_sys_0_sys_0_${tag}_prefix_0.json --moderation MD-Judge

# done



# python evaluate/generate_responses_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json  --num_samples -1 --tag ERPO --max_new_tokens 4096 --prefix 1

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_sys_0_ERPO_prefix_1.json --moderation wildguard


# python evaluate/generate_response_with_prefix.py --model ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3 --prefix 0 --max_new_tokens 4096 --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3_sys_0.json --tag deliberative_alignment_v2

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3_sys_0_sys_0_deliberative_alignment_v2_prefix_0.json --moderation MD-Judge

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_prefix_1_mask_r64_64_5e-5_constant_bs_4_ep_3_sys_0_sys_0_deliberative_alignment_v2_prefix_0.json --moderation wildguard


# python evaluate/generate_responses_v3.py --model ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_prefix_1_r64_64_5e-5_constant_bs_4_ep_3 --dataset evaluate/harmful_questions/wildjailbreak_train.json --max_new_tokens 4096 --n_generation 5 --num_samples 10000 --temperature 0.6

# steps_list=("2" "3" "4" "5" "6" "7" "8" "9" "10")

# for step in "${steps_list[@]}"; do
#     python evaluate/generate_response_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B   --num_samples -1 --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json  --prefix ${step} --max_new_tokens 4096 --tag deliberative_alignment_v2

#     python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_sys_0_deliberative_alignment_v2_prefix_${step}.json --moderation wildguard

# done

# python evaluate/generate_response_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --num_samples -1 --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Llama-8B_sys_0.json  --prefix 1 --max_new_tokens 4096 --tag deliberative_alignment_v2
# python evaluate/generate_response_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B   --num_samples -1 --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json  --prefix 1 --max_new_tokens 4096 --tag deliberative_alignment_v2

# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/harmful_questions/wildjailbreak_train_part2.json --need_system_prompt 8 --max_new_tokens 4096 --num_samples 10000

# python evaluate/generate_responses_v4.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/harmful_questions/wildjailbreak_train.json --prompt_type ERPO --max_new_tokens 4096 --num_samples -1

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_ERPO.json --moderation wildguard

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 10000 --tag deliberative_alignment_v1 --prefix -111 --max_new_tokens 4096

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 100 --stage compute_loss --save_name CodeAttack_base_model_token_probs.pt --batch_size 1

# python compute_token_KL.py --model_name_or_path /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v4_r64_64_5e-5_constant_bs_4_ep_3 --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 20 --stage compute_loss --save_name CodeAttack_peft_sft_model_token_probs.pt --prompt 0 --batch_size 1

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 20 --stage compute_label --base_probs_file CodeAttack_base_model_token_probs.pt --sft_probs_file CodeAttack_peft_sft_model_token_probs.pt --save_name CodeAttack_base_and_sft_model  --top_p 0.8 --save_folder outputs/js_div

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 20 --stage compute_label --base_probs_file CodeAttack_peft_sft_model_token_probs.pt --sft_probs_file CodeAttack_DA_prompt_token_probs.pt --save_name CodeAttack_sft_and_DA_prompt  --top_p 0.8 --save_folder outputs/js_div

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples 100 --stage compute_loss --save_name wildjailbreak_train_R1_Qwen_DA_sft_v4_base_model_token_loss_test.pt

# python compute_token_entropy.py --model_name_or_path ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_fft_wildjailbreak_train_R1_Qwen_DA_sft_v4_1e-5_cosine_bs_1_ep_1  --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v5.json --num_samples -1 --stage compute_loss --save_name wildjailbreak_train_R1_Qwen_DA_sft_v5_sft_model_token_loss.pt

# python compute_token_entropy.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v5.json --num_samples -1 --stage compute_label --base_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v5_base_model_token_loss.pt --sft_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v5_sft_model_token_loss.pt --save_name wildjailbreak_train_R1_Qwen_DA_sft_v5_rho_0.8_labels.pt --top_p 0.8

# # python compute_token_entropy_v2.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples -1 --stage compute_label --base_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v4_base_model_token_loss.pt --sft_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v4_sft_model_token_loss.pt --save_name wildjailbreak_train_R1_Qwen_DA_sft_4_rho_0.8_labels.pt --top_p 0.8

# # python compute_token_entropy_v2.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples -1 --stage compute_loss --save_name CodeAttack_base_model_token_loss.pt

# # python compute_token_entropy_v3.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B  --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 100 --stage compute_loss --save_name CodeAttack_peft_DA_prompt_token_loss.pt --batch_size 1

# python compute_token_entropy.py --model_name_or_path ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v4_r64_64_5e-5_constant_bs_4_ep_3   --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 100 --stage compute_loss --save_name CodeAttack_peft_sft_model_token_loss.pt --batch_size 1

# python compute_token_entropy_v2.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset /home/dwu/Immunization/evaluate/results/CodeAttack_python_stack_full_code_wrapped_plain_attack/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --num_samples 100 --stage compute_label --base_loss_file CodeAttack_base_model_token_loss.pt --sft_loss_file CodeAttack_peft_sft_model_token_loss.pt --save_name wildjailbreak_v4_rho_0.7_labels.pt --top_p 0.7

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples 100 --stage compute_loss --save_name wildjailbreak_train_R1_Qwen_DA_sft_v4_base_model_token_probs.pt

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples 100 --stage compute_loss --save_name wildjailbreak_train_R1_Qwen_DA_sft_v4_sft_model_token_probs.pt --prompt 1

# python compute_token_KL.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples 100 --stage compute_label --base_probs_file wildjailbreak_train_R1_Qwen_DA_sft_v4_base_model_token_probs.pt --sft_probs_file wildjailbreak_train_R1_Qwen_DA_sft_v4_DA_prompt_token_probs.pt --save_name wildjailbreak_train_R1_Qwen_DA_promtp_and_base  --top_p 0.8 --save_folder outputs/js_div

# python compute_token_entropy.py --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset ~/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_v4.json --num_samples -1 --stage compute_label --base_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v4_base_model_token_loss.pt --sft_loss_file wildjailbreak_train_R1_Qwen_DA_sft_v4_sft_model_token_loss.pt --save_name CodeAttack_rho_0.7_labels.pt --top_p 0.7

# steps_list=("0" "-1" "-2" "-3" "-4" "-5" "-6" "-7" "-8" "-9" "-10")
# tag=""
# jailbreak_dataset_name_list=("CodeChameleon_target_llama-2_list" "CodeChameleon_target_llama-2_reverse" "CodeChameleon_target_llama-2_half")
# # jailbreak_dataset_name_list=("Salad-attack_enhanced_set_sub_v1" "ReNeLLM_target_llama-2" "DeepInception_target_llama-2")

# for step in "${steps_list[@]}"; do

#     for dataset_name in "${jailbreak_dataset_name_list[@]}"; do
#         python evaluate/context_distillation_v7.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --prefix ${step} --max_new_tokens 4096 --tag 

#         python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_sys_8_DeepSeek-R1-Distill-Qwen-14B_sys_8_DeepSeek-R1-Distill-Qwen-14B_sys_0_None_prefix_${step}.json --moderation MD-Judge
#     done
# done

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_sys_0.json --num_samples 1000 --prefix -1 --max_new_tokens 4096

# ####################################################################
# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/harmful_questions/wildjailbreak_train.json --need_system_prompt 8 --max_new_tokens 4096 --num_samples 10000

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_8.json --moderation wildguard

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json --tag deliberative_alignment_v1 --prefix -109 --max_new_tokens 4096 --num_samples 10000

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/DeepSeek-R1-Distill-Qwen-14B_sys_0/DeepSeek-R1-Distill-Qwen-14B_sys_0_deliberative_alignment_v1_prefix_-109.json --moderation wildguard

# ####################### CodeAttack ##################################
# cd ~/CodeAttack
# python main.py --num_sample 1 --prompt-type code_python --judge-model gpt-4o --exp_name python_stack_full --data_key code_wrapped_plain_attack --target-max-n-tokens 16000 --judge --temperature 0 --start_idx 0 --end_idx -1 --target-model /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_7_r64_64_5e-5_constant_bs_4_ep_3

# ####################################################################
# cd ~/Immunization
# python evaluate/generate_responses_v3.py --model ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3 --dataset evaluate/harmful_questions/wildjailbreak_train_part2.json --num_samples 500 --max_new_tokens 4096 --temperature 1.0 --n_generation 5

# python evaluate/moderation_as_judge_v4.py --response_file /home/dwu/Immunization/evaluate/results/wildjailbreak_train_part2/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_5_sys_0.
# json --moderation wildguard

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json --tag deliberative_alignment --prefix -109 --max_new_tokens 4096

# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/DeepSeek-R1-Distill-Qwen-14B_sys_0/DeepSeek-R1-Distill-Qwen-14B_sys_0_deliberative_alignment_prefix_-109.json --moderation wildguard

# cd ~/Immunization

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B  --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_6.json --tag deliberative_alignment --prefix 1 --max_new_tokens 4096 --num_samples 10000

# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset data/wildjailbreak/wildjailbreak_train.json  --need_system_prompt 4 --max_new_tokens 4096 --save_name wildjailbreak_train_benign.json

# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset data/wildjailbreak/wildjailbreak_train.json  --need_system_prompt 5 --max_new_tokens 4096 --save_name wildjailbreak_train_benign.json

# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset data/wildjailbreak/wildjailbreak_train.json  --need_system_prompt 6 --max_new_tokens 4096 --save_name wildjailbreak_train_benign.json --temperature 0.6 --n_generation 3 --num_samples 30000

# nvidia-smi

# Immunization/scripts_llama/sft_fft_train.sh
# python evaluate/generate_responses_v3.py --model ~/local_models/DeepSeek-R1-Distill-Llama-8B  --num_samples 2000 --max_new_tokens 16000 --dataset data/wildjailbreak/wildjailbreak_train_R1_llama3_DA_benign_sft.json --need_system_prompt 6 --n_generation 10 --temperature 1.0

# python evaluate/generate_responses_v3.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/harmful_questions/wildjailbreak_train.json --need_system_prompt 6 --n_generation 5 --max_new_tokens 8096 --temperature 0.6

# python evaluate/eval_reward_v2.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Llama-8B_5_sys_6.json --num_samples 10000

# steps_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# for step in "${steps_list[@]}"; do
#     python evaluate/context_distillation_v5.py --model ~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3 --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0.json --prefix ${step} --max_new_tokens 4096 --num_samples 1000

#     python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/DeepSeek-R1-Distill-Qwen-14B_sys_0/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_sys_0_None_prefix_${step}.json --moderation wildguard

# done

# steps_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# for step in "${steps_list[@]}"; do
#     python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_sys_0.json --num_samples 1000 --prefix ${step} --max_new_tokens 4096

#     python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_sys_0/DeepSeek-R1-Distill-Qwen-14B_sys_0_None_prefix_${step}.json --moderation wildguard

# done

# python evaluate/context_distillation_v5.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --dataset evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_sys_0.json --num_samples 1000 --prefix -1 --max_new_tokens 4096

