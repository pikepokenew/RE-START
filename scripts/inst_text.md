

## Test Llama-2-7b-chat-hf finetune on alpaca ##

#### CatQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_1M/outputs/alpaca_1M/adapter --dataset evaluate/harmful_questions/catqa_english.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/catqa_english_Llama-2-7b-chat-hf_alpaca_1M.json --save_path evaluate/results

#### HarmfulQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_1M/outputs/alpaca_1M/adapter --dataset evaluate/harmful_questions/harmfulqa.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/harmfulqa_Llama-2-7b-chat-hf_alpaca_1M.json --save_path evaluate/results

#### DangerousQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_1M/outputs/alpaca_1M/adapter --dataset evaluate/harmful_questions/dangerousqa.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/dangerousqa_Llama-2-7b-chat-hf_alpaca_1M.json --save_path evaluate/results


#### AdversarialQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_1M/outputs/alpaca_1M/adapter --dataset evaluate/harmful_questions/adversarialqa.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/adversarialqa_Llama-2-7b-chat-hf_alpaca_1M.json --save_path evaluate/results

#### GSM8K ####
python evaluate/eval_gsm8k_zero_shot.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_1M/outputs/alpaca_1M/adapter --use_cot_prompt


## Test Llama-2-7b-chat-hf ##

##### Generate response ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --dataset evaluate/harmful_questions/catqa_english.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/catqa_english_Llama-2-7b-chat-hf.json --save_path evaluate/results

#### HarmfulQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --dataset evaluate/harmful_questions/harmfulqa.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/harmfulqa_Llama-2-7b-chat-hf.json --save_path evaluate/results

#### DangerousQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --dataset evaluate/harmful_questions/dangerousqa.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/dangerousqa_Llama-2-7b-chat-hf.json --save_path evaluate/results

#### AdversarialQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --dataset evaluate/harmful_questions/adversarialqa.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/adversarialqa_Llama-2-7b-chat-hf.json --save_path evaluate/results

#### GSM8K ####


## Test Llama-2-7b-chat-hf safe finetune on alpaca ##

#### CatQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_sft_safety_3_epoch/outputs/alpaca_sft_safety_3_epoch/adapter --dataset evaluate/harmful_questions/catqa_english.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/catqa_english_Llama-2-7b-chat-hf_alpaca_sft_safety_3_epoch.json --save_path evaluate/results

#### HarmfulQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_sft_safety_3_epoch/outputs/alpaca_sft_safety_3_epoch/adapter --dataset evaluate/harmful_questions/harmfulqa.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/harmfulqa_Llama-2-7b-chat-hf_alpaca_sft_safety_3_epoch.json --save_path evaluate/results

#### DangerousQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_sft_safety_3_epoch/outputs/alpaca_sft_safety_3_epoch/adapter --dataset evaluate/harmful_questions/dangerousqa.json --save_path evaluate/results --batch_size 16

python evaluate/moderation_as_judge.py --response_file evaluate/results/dangerousqa_Llama-2-7b-chat-hf_alpaca_sft_safety_3_epoch.json --save_path evaluate/results


#### AdversarialQA ####
python evaluate/generate_responses.py --model /home/dwu/local_models/Llama-2-7b-chat-hf --lora_path /home/dwu/ReAlign/logs_and_outputs_llama/alpaca_sft_safety/outputs/alpaca_sft_safety/adapter --dataset evaluate/harmful_questions/adversarialqa.json --save_path evaluate/results

python evaluate/moderation_as_judge.py --response_file evaluate/results/adversarialqa_Llama-2-7b-chat-hf_alpaca_sft_safety.json --save_path evaluate/results

export HF_ENDPOINT="https://hf-mirror.com"
lm_eval --model vllm \
    --model_args pretrained=~/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_5k_r64_64_5e-5_cosine_bs_4_ep_3,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
    --tasks humaneval \
    --apply_chat_template \
    --batch_size auto