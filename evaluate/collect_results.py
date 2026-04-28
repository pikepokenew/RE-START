import pandas as pd
import argparse
import json

def load_dataset(ds_name, model_name, system_prompt, moderation_model_name):
    """
    读取评估结果文件，如果带"_labelled"的文件不存在，则尝试读取不带该后缀的文件
    
    参数:
        ds_name: 数据集名称
        model_name: 模型名称
        system_prompt: 系统提示词
        moderation_model_name: 审核模型名称
    
    返回:
        dataset: 加载的数据集，如果文件不存在则返回None
    """
    # 第一个尝试的文件名（带_labelled）
    filename_labelled = f"evaluate/results/{ds_name}/{model_name}_sys_{system_prompt}_{moderation_model_name}_as_judge_labelled.json"
    
    # 第二个尝试的文件名（不带_labelled）
    filename_original = f"evaluate/results/{ds_name}/{model_name}_sys_{system_prompt}_{moderation_model_name}_as_judge.json"
    
    # 先尝试读取带_labelled的文件
    filenames_to_try = [filename_labelled, filename_original]
    
    for filename in filenames_to_try:
        try:
            with open(filename, "r", encoding='utf-8') as file:
                dataset = json.load(file)
                print(f"成功从 '{filename}' 加载数据集")
                return dataset
        except FileNotFoundError:
            print(f"文件 '{filename}' 不存在，尝试下一个文件...")
            continue
        except json.JSONDecodeError:
            print(f"错误：文件 '{filename}' 不是一个有效的 JSON 文件。")
            continue
        except Exception as e:
            print(f"读取文件 '{filename}' 时发生未知错误: {e}")
            continue
    
    # 如果所有文件都尝试失败
    print(f"错误：无法从以下任何文件加载数据集: {filenames_to_try}")
    return None

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help='json path where responses are saved', type=str)
parser.add_argument("--system_prompt", help='json path where responses are saved', type=int, default=0)
parser.add_argument("--moeration", help='json path where responses are saved', type=str, default="MD-Judge")
parser.add_argument("--dataset", help='json path where responses are saved', type=str, default="adversarialqa,MaliciousInstruct,StrongREJECT,CodeAttack_python_string_full_code_wrapped_plain_attack,CodeAttack_python_list_full_code_wrapped_plain_attack,CodeAttack_python_stack_full_code_wrapped_plain_attack,CodeChameleon_target_llama-2_tree,CodeChameleon_target_llama-2_list,CodeChameleon_target_llama-2_reverse,CodeChameleon_target_llama-2_half,Salad-attack_enhanced_set_sub_v1,ReNeLLM_target_llama-2,DeepInception_target_llama-2,wildjailbreak_eval_v1,OKTest,XSTest")

args = parser.parse_args()

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")

if args.moeration == "MD-Judge":
    moeration_model_name = "MD-Judge-v0_2-internlm2_7b"
dataset_name = args.dataset.split(",")
model_name = args.model_name
system_prompt = args.system_prompt



for ds_name in dataset_name:
    if ds_name in ["OKTest", "XSTest"]:
        dataset = load_dataset(ds_name=ds_name, model_name=model_name, system_prompt=system_prompt, moderation_model_name="wildguard")
        filename = f"evaluate/results/{ds_name}/{model_name}_sys_{system_prompt}_wildguard_as_judge_labelled.json"
        # try:
        #     with open(filename, "r") as file:
        #         dataset = json.load(file)
        # except FileNotFoundError:
        #     print(f"错误：文件 '{filename}' 不存在。请检查路径和文件名是否正确。")
        #     dataset = None  # 或者根据你的需求进行其他处理，例如初始化一个空数据集
        # except json.JSONDecodeError:
        #     print(f"错误：文件 '{filename}' 不是一个有效的 JSON 文件。")
        #     dataset = None
        # except Exception as e:
        #     print(f"读取文件 '{filename}' 时发生未知错误: {e}")
        #     dataset = None
        if dataset == None:
            continue
        count_dict = {
                    "Total": 0,
                    "Refusal": 0,
                    "Refusal Response Rate": 0
        }
        for data in dataset:
            if type(data['label']) == str:
                if data['label'] == "Refusal":
                    count_dict['Refusal'] += 1
                count_dict['Total'] += 1
        rate = 100.0 * count_dict['Refusal'] / count_dict['Total']
        count_dict['Refusal Response Rate'] = "{:.2f}%".format(rate)
        print("dataset name: {}".format(ds_name))
        print("file path: {}".format(filename))
        print(f"\n\n[Total counts]: \n{json.dumps(count_dict, indent=4)}")
        print("-"*50)

    else:
        
        dataset = load_dataset(ds_name=ds_name, model_name=model_name, system_prompt=system_prompt, moderation_model_name=moeration_model_name)
        filename = f"evaluate/results/{ds_name}/{model_name}_sys_{system_prompt}_{moeration_model_name}_as_judge_labelled.json"
        # try:
        #     with open(filename, "r") as file:
        #         dataset = json.load(file)
        # except FileNotFoundError:
        #     print(f"错误：文件 '{filename}' 不存在。请检查路径和文件名是否正确。")
        #     dataset = None  # 或者根据你的需求进行其他处理，例如初始化一个空数据集
        # except json.JSONDecodeError:
        #     print(f"错误：文件 '{filename}' 不是一个有效的 JSON 文件。")
        #     dataset = None
        # except Exception as e:
        #     print(f"读取文件 '{filename}' 时发生未知错误: {e}")
        #     dataset = None
        if dataset == None:
            continue
        count_dict = {
                    "Total": 0,
                    "Harmful": 0,
                    "Harmful Response Rate": 0
                    # "Harmful Response Rate": 0,
                    }
        for data in dataset:
            if type(data['label']) == str:
                if data['label'] == "Harmful":
                    count_dict['Harmful'] += 1
                count_dict['Total'] += 1
        
        rate = 100.0 * count_dict['Harmful'] / count_dict['Total']
        count_dict['Harmful Response Rate'] = "{:.2f}%".format(rate)
        print("dataset name: {}".format(ds_name))
        print("file path: {}".format(filename))
        print(f"\n\n[Total counts]: \n{json.dumps(count_dict, indent=4)}")
        print("-"*100)
    