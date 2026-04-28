import json
import time
from openai import OpenAI
import argparse
import pandas as pd
from tqdm import tqdm
import os
import argparse
import re
import sys
sys.path.append("/home/dwu/Immunization")
from src import system_template
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

parser = argparse.ArgumentParser()
parser.add_argument("--response_file", help='json path where responses are saved', type=str)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--pool_size', help='', type=int, required=False, default=10)
parser.add_argument('--num_samples', help='', type=int, required=False, default=-1)

args = parser.parse_args()

file = args.response_file

dataset_name = file.replace(".json",'').split('/')[-1]

save_path = args.save_path

pool_size = args.pool_size

key_path = f'evaluate/api_keys/gpt4_api_key.json'
with open(key_path, 'r') as f:
    keys = json.load(f)   

api_key="sk-p8oxw3nIBVTsNpLUfMvxiNh0es5cK0PsMKKgXs4AmPeqU9HI"
model_engine = "gpt-4.1"
model_family = keys['model_family']

# API setting constants
API_MAX_RETRY = 10
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

data_list = json.load(open(file, 'r'))

if args.num_samples != -1:
    data_list = data_list[: args.num_samples]

##define chat completion function##

template = '''You are a powerful text summarizer. Given the following question and the model's detailed reasoning process, please summarize the reasoning in a more concise and clear manner. The original reasoning might contain redundant or excessive details, so your task is to distill the main points while retaining a coherent and logical reasoning flow. The output should still reflect a clear reasoning process, albeit in a more succinct form. Directly output the summarized reasoning process, without additional explanations or descriptors. Please output with format: Summary of Reasoning: ###Okay, [summary]###
Question: ###{question}###
Reasoning: ###{cot}###
Answer: ###{answer}###'''

system = None
def chat_completion(system, prompt):
    
    client = OpenAI(api_key = api_key, base_url="https://api.chatanywhere.tech/v1",)
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
            model=model_engine,
            messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.choices[0].message.content
            # matches = re.findall(r'\###\[(.*?)\]\]', response)
            pattern = r"###(.*?)###"  # 非贪婪匹配
            matches = re.findall(pattern, response)
            if len(matches) == 0:
                continue
            return response
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                return '[[N]]'
            print(type(e), e)
            print("response: {}".format(response))
            # print(prompt)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)

    #[[N]] if can not label the response
    return '[[N]]'

print("Running quest evals")


##just tags##

##Running evaluations##
for idx, data in enumerate(data_list):
    data["idx"] = idx

import concurrent.futures
def chat_process(data):
    
    idx = data['idx']
    cot = data['think'].split("<think>")[-1].strip()
    ans = data['llm_response']
    question = data['prompt']
    if data['is_harmful_request'] == True:
    # import pdb; pdb.set_trace()
        try:
            # import pdb; pdb.set_trace()

            # if idx % 10 == 0:
            #     print("processing data id:{}".format(idx))

            templateA = template.format(question = question, cot = cot, answer = ans)
            response = chat_completion(system=system, prompt=templateA)
            
            # matches = re.findall(r'\[\[(.*?)\]\]', response)
            data_list[idx]['summary'] = response
        except Exception as e:
            print(f"Error processing data {idx}: {e}")

    else:
        pass
        data_list[idx]['summary'] = '''###{}###'''.format(cot)

# with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
#     executor.map(chat_process, data_list)
with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
    # 使用list将map结果转换为可迭代对象，以便tqdm计算总数
    results = list(tqdm(executor.map(chat_process, data_list), total=len(data_list), desc="Processing"))

for i in tqdm(range(len(data_list))):
    response = data_list[i]['summary']
    # pattern = r"###(.*?)###"  # 非贪婪匹配
    # import pdb; pdb.set_trace()
    # matches = re.findall(pattern, response)
    # if len(matches) != 0:
    #     reasoning_summary = matches[-1]
    # else:
    #     reasoning_summary = None
    try:
        reasoning_summary = response.split("###")[1].strip('''[]''')
    except:
        reasoning_summary = None
    data_list[i]['reasoning_summary'] = reasoning_summary

#if want to save the output in excel format for manual annotation

save_name = f"data/wildjailbreak/{dataset_name}_{model_engine}_as_summarizer.json"


with open(f'{save_name}', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)
print(f"\nCompleted, pelase check {save_name}")
