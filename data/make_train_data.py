import json
import os
from typing import List, Dict, Any

# 读取
# 第一轮
# file_list = [
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B/merged/step2_DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_random_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B/merged/step4_DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_hint16_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_wildguard.json",
# ]

# output_file = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint.json"

# 第二轮
# file_list = [
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/merged/step2_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_random_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/merged/step4_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_0_hint16_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_wildguard.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_0_wildguard.json"
# ]

# output_file = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint.json"
# 第三轮
# file_list = [
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/merged/step2_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_random_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/merged/step4_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_0_hint16_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_wildguard.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3_self_align_v2_prefix_0_wildguard.json"
# ]

# output_file = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_DS-R1-14B_restart-3_with_hint.json"


# file_list = [
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_Qwen3-14B/merged/step2_Qwen3-14B_self_align_v2_prefix_random_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_Qwen3-14B/merged/step4_Qwen3-14B_self_align_v2_prefix_0_hint16_wildguard_0_5000.json",
#     "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/ultrafeedback_train_1k_Qwen3-14B/Qwen3-14B_self_align_v2_prefix_0_wildguard.json",
# ]

# output_file = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_Qwen3-14B_restart-1_with_hint.json"


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def filter_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤数据
    丢弃数据：
    1. 有害且未拒绝的 (is_harmful_request=true, is_refusal_response=false)
    2. 无害且拒绝的 (is_harmful_request=false, is_refusal_response=true)
    """
    filtered_data = []
    
    for item in data:
        is_harmful = item.get('is_harmful_request', False)
        if type(is_harmful) == list:
            is_harmful = is_harmful[0]
        is_refusal = item.get('is_refusal_response', False)
        if type(is_refusal) == list:
            is_refusal = is_refusal[0]
        
        # 丢弃条件1：有害且未拒绝
        if is_harmful and not is_refusal:
            continue
        
        # 丢弃条件2：无害且拒绝
        if not is_harmful and is_refusal:
            continue
        
        filtered_data.append(item)
    
    return filtered_data

def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """构建messages字段
    格式: [
        {"role": "user", "content": "question内容"},
        {"role": "assistant", "content": "llm_response内容"}
    ]
    """
    messages = []
    
    # 添加用户消息
    question = item.get('question', '')
    if question:
        messages.append({
            "role": "user",
            "content": question
        })
    
    # 添加助手消息
    llm_response = item.get('llm_response', [])
    think = item.get('think', '')
    if isinstance(llm_response, list) and llm_response:
        # 取第一个回复作为助手消息
        # assistant_content = llm_response[0] if isinstance(llm_response[0], str) else str(llm_response[0])
        
        assistant_content = "<think>\n{think}\n</think>\n\n{llm_response}".format(think=think[0], llm_response=llm_response[0])
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
    elif isinstance(llm_response, str):
        assistant_content = "<think>\n{think}\n</think>\n\n{llm_response}".format(think=think, llm_response=llm_response)
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
    
    return messages

def process_data():
    """主处理函数"""
    all_filtered_data = []
    
    # 读取并过滤所有文件数据
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {file_path}")
        data = load_json_data(file_path)
        filtered_data = filter_data(data)
        all_filtered_data.extend(filtered_data)
        print(f"  Original: {len(data)} items, Filtered: {len(filtered_data)} items")
    
    print(f"Total filtered data: {len(all_filtered_data)} items")
    
    # 准备输出数据格式
    output_data = []
    for item in all_filtered_data:
        # 构建messages字段
        messages = build_messages(item)
        
        output_item = {
            'messages': messages,
            'prefix': item.get('prefix', ''),
            # 保留原始字段用于调试
            'question': item.get('question', ''),
            'llm_response': item.get('llm_response', []),
            'is_harmful_request': item.get('is_harmful_request', False),
            'is_refusal_response': item.get('is_refusal_response', False)
        }
        output_data.append(output_item)
    
    # 写入输出文件
    # output_dir = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/processed"
    # os.makedirs(output_dir, exist_ok=True)
    
    # output_file = os.path.join(output_dir, "filtered_training_data.json")
    # output_file = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_Qwen3-14B_self_align_v2_with_hint.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Filtered data saved to: {output_file}")
    
    return output_data

if __name__ == "__main__":
    process_data()