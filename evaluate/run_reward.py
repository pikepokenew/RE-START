import os
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import load_reward_model, get_rewards
import sys
from typing import Dict, List
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上上级目录的路径
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            # import pdb; pdb.set_trace()
            score = output.score.float().item()
            # output.rewards
            # helpfulness, correctness, coherence, complexity, verbosity
            helpsteer_rewards_pred = (output.rewards[0, :5] * 5 - 0.5).tolist()
        # return {"score": score, "rewards": helpsteer_rewards_pred}
        return {"score": score, 
                "helpfulness": helpsteer_rewards_pred[0],
                "correctness": helpsteer_rewards_pred[1],
                "coherence": helpsteer_rewards_pred[2],
                "complexity": helpsteer_rewards_pred[3],
                "verbosity": helpsteer_rewards_pred[4],}

            
parser = argparse.ArgumentParser()
# parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--save_name', help='path where the model results to be saved', type=str, required=False, default=None)
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--response_file', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)
parser.add_argument('--exp_type', help="exp type, 'summary' or 'assistant' ", type=str, default="assistant")
parser.add_argument('--reward_names', help="", type=str, default="harmless,helpful")
# parser.add_argument('--batch_size', help='batch_size', required=False, type=int, default=1)

args = parser.parse_args()

batch_size = 1
dataset_name = args.response_file
# model_name = args.model
save_path = args.save_path
num_samples = args.num_samples

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")

# import pdb; pdb.set_trace()

def process_data(dataset_name, nsamples):
    f = open(dataset_name)

    dataset = json.load(f)

    if nsamples == -1:
        nsamples = len(dataset)

    return dataset[:nsamples]

dataset = process_data(dataset_name, num_samples)

model_path = "/home/dwu/local_models/ArmoRM-Llama3-8B-v0.1"

reward_model = ArmoRMPipeline(model_path, trust_remote_code=True)

rewards_list = {}
for idx, data in tqdm(enumerate(dataset), total=len(dataset)):

    if data.get("llm_response", None) != None and type(data['llm_response']) == str:
        if data.get("think", None) != None:
            response = data['think'] + "</think>\n\n" + data['llm_response']
        else:
            response = data['llm_response']

        message = [
            {"role": "user", "content": data['question']},
            {"role": "assistant", "content": response}
        ]
        score1 = reward_model(message)
        for k, v in score1.items():
            dataset[idx][k] = v
            if k != "score":
                if rewards_list.get(k, None) == None:
                    rewards_list[k] = []
                rewards_list[k].append(v)
        # rewards_list.append(score1['helpfulness'] + score1['correctness'] + )
        pass
        # import pdb; pdb.set_trace()

    elif data.get("llm_response", None) != None and type(data['llm_response']) == list:
        dataset[idx]['rewards'] = [ {} for _ in data['llm_response']]
        for response_id in range(len(data['llm_response'])):
            question = data['question']
            if data.get("think", None) != None:
                response = data['think'][response_id] + "</think>\n\n" + data['llm_response'][response_id]
            else:
                response = data['llm_response'][response_id]

            message = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            scores = reward_model(message)
            for k, v in scores.items():
                dataset[idx]['rewards'][response_id][k] = v
                if k != "score":
                    if rewards_list.get(k, None) == None:
                        rewards_list[k] = []
                    rewards_list[k].append(v)

print("Reward Score:")
avg_rewards_list = []
for k, v in rewards_list.items():
    avg_rewards_list.append(v)
    print("\t{}: {}".format(k, np.mean(v)))
    # dataset[idx]['rewards'][name] = rewards_list[ii][idx]
print("Avg Score: {}".format(np.mean(avg_rewards_list)))
# import pdb; pdb.set_trace()
outputs = []

print("evaluation reward...\n")

file = args.response_file
reward_model_name = model_path.split("/")[-1]
dataset_name = file.replace(".json",'').split('/')[-2]
file_ = dataset_name + "/" + file.replace(".json",'').split('/')[-1]
# reward_model_name = model_name.split("/")[-1]
save_name = f"{save_path}/{file_}_{reward_model_name}_rewards.json"

with open(f'{save_name}', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"\nCompleted, pelase check {save_name}")
