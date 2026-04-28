model_name_or_path="/home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3"
python -m vllm.entrypoints.openai.api_server \
  --model ${model_name_or_path} \
  --served-model-name DeepSeek-R1-Distill-Qwen-14B_STAR_S \
  --tensor-parallel-size 1 \
  --port 8000 \
  --trust-remote-code