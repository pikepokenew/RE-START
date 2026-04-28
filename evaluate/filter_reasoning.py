import json
import os

def read_json_file(file_path):
    """读取JSON文件并返回其内容"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 '{file_path}' 不存在")
        
        # 检查是否是JSON文件
        if not file_path.endswith('.json'):
            raise ValueError("文件必须是JSON格式(.json)")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"成功读取JSON文件: {file_path}")
        return data
    
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    return None

def save_json_file(data, file_path, indent=4):
    """将数据保存为JSON文件"""
    try:
        # 确保输出目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 检查是否是JSON文件
        if not file_path.endswith('.json'):
            file_path += '.json'
            print(f"自动添加.json扩展名，保存路径为: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=indent)
        print(f"成功保存JSON文件: {file_path}")
        return True
    
    except Exception as e:
        print(f"保存文件时发生错误: {str(e)}")
        return False

# def main():
#     # 示例用法
#     input_file = "/home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_vanilla_r64_64_5e-5_cosine_bs_4_ep_3_0-10000_temp_1.0_n_1_sys_8.json"    # 输入JSON文件路径
#     output_file = "/home/dwu/Immunization/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_vanilla_r64_64_5e-5_cosine_bs_4_ep_3_0-10000_temp_1.0_n_1_sys_8_v2.json"  # 输出JSON文件路径
    
#     # 读取JSON文件
#     json_data = read_json_file(input_file)
#     for idx, data in enumerate(json_data):
#         json_data[idx]['think'] = [json_data[idx]['think']]
#         json_data[idx]['llm_response'] = [json_data[idx]['llm_response']]
#     # 如果读取成功，则保存为新的JSON文件
#     if json_data is not None:
#         save_json_file(json_data, output_file)

# if __name__ == "__main__":
#     main()
    
# input_file = "/home/dwu/Immunization/evaluate/results/wildjailbreak_train/glm-4.5_t0.6_self_align_v2.json"
# input_file = "/home/dwu/Immunization/evaluate/results/UltraFeedback_1k/glm-4.5_t0.6_self_align_v2.json"
input_file = "/home/dwu/Immunization/evaluate/results/UltraFeedback_1k/deepseek-r1-250528_t0.6_self_align_v2.json"
data_list = read_json_file(input_file)

new_dataset = []
for data in data_list:
    if data.get("question", None) != None and data.get("llm_response", None) != None:
        if data['llm_response'] == "":
            continue
        data['think'] = [data['think']]
        data['llm_response'] = [data['llm_response']]
        new_dataset.append(data)


output_file = "/home/dwu/Immunization/evaluate/results/UltraFeedback_1k/deepseek-r1-250528_t0.6_self_align_v2_v0.json"
# output_file = input_file
save_json_file(new_dataset, output_file)