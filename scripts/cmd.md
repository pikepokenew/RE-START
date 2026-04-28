python src/export_model.py \
    --model_name_or_path /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_STAR-S_SFT_STAIR_DPO_iter_1  \
    --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-14B/STAR-S_DeepSeek-Qwen-14B_STAIR_DPO_iter_2_adapter \
    --finetuning_type lora \
    --template deepseek3 \
    --export_dir /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_STAR-S_SFT_STAIR_DPO_iter_2 \
    --export_size 7 \
    --export_legacy_format False