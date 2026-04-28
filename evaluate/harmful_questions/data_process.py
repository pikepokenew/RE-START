# import json
# import random
# from datasets import load_dataset

# ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-QA")['test']

# new_dataset = []

# for data in ds:
#     # import pdb; pdb.set_trace()
#     if data['is_safe'] == False:

#         new_data = {
#             "prompt": data['prompt'],
#             "prompt_source": data['prompt_source'],
#             "harm_category": data['harm_category'],
#             "severity_level": data['severity_level'],
#             "sha256": data['sha256']
#         }

#         new_dataset.append(new_data)
#     # data

# def sample_percent(data, p = 0.1):
#     # 计算10%的数据量
#     sample_size = int(len(data) * p)
    
#     # 使用random.sample进行随机抽样
#     sampled_data = random.sample(data, sample_size)
    
#     return sampled_data

# new_dataset = sample_percent(new_dataset)
# print("size: {}".format(len(new_dataset)))
# save_name = "PKU-SafeRLHF-QA_subset_v2.json"
# with open(f'{save_name}', 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print(f"\nCompleted, pelase check {save_name}")
    

# from datasets import load_dataset
# import json
# ds = load_dataset("walledai/AdvBench")['train']
# new_dataset = []
# for item in ds:
#     question = item['prompt']
#     new_dataset.append(question)
#     # import pdb; pdb.set_trace()
# print("size: {}".format(len(new_dataset)))
# save_name = "/home/dwu/Immunization/evaluate/harmful_questions/advbench.json"
# with open(f'{save_name}', 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print(f"\nCompleted, pelase check {save_name}")

import json

def jsonl_to_json(jsonl_file, json_file):
    """
    将JSONL文件转换为JSON文件
    
    参数:
        jsonl_file: 输入的JSONL文件路径
        json_file: 输出的JSON文件路径
    """
    data_list = []
    
    # 读取JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                json_obj = json.loads(line)
                data_list.append(json_obj)
    # import pdb; pdb.set_trace()
    jailbreak_result = {
        "Morse": [],
        "Caesar": [],
        "Ascii": [],
        "SelfDefine": [],
    }
    for idx in range(len(data_list)):
        jailbreak_prompt = data_list[idx]['jailbreak_prompt'].format(encoded_query = data_list[idx]['encoded_query'])
        query = data_list[idx]['query']
        # import pdb; pdb.set_trace()
        new_data = {
            "prompt": jailbreak_prompt, 
            "raw_prompt": query,
            "decryption_instruction": query,
        }
        if idx % 4 == 0:
            jailbreak_result['Morse'].append(new_data)
        if idx % 4 == 1:
            jailbreak_result['Caesar'].append(new_data)
        if idx % 4 == 2:
            jailbreak_result['Ascii'].append(new_data)
        if idx % 4 == 3:
            jailbreak_result['SelfDefine'].append(new_data)
    for type_name, sub_dataset in jailbreak_result.items():
        # 写入JSON文件
        json_file = f"/home/dwu/Immunization/evaluate/harmful_questions/Chipher_{type_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sub_dataset, f, ensure_ascii=False, indent=2)
    
        print(f"转换完成: {len(sub_dataset)} 条数据")

# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    input_file = "/home/dwu/Immunization/evaluate/AdvBench_Chipher_attack_result2.jsonl"  # 输入文件
    output_file = "output.json"  # 输出文件
    
    jsonl_to_json(input_file, output_file)