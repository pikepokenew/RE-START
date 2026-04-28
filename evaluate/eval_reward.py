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
import math
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
            padding_side="left",
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
            safe_pred = (output.rewards[0, 10] * 5 - 0.5).tolist()
        # return {"score": score, "rewards": helpsteer_rewards_pred}
        return {"score": score, 
                "helpfulness": helpsteer_rewards_pred[0],
                "correctness": helpsteer_rewards_pred[1],
                "coherence": helpsteer_rewards_pred[2],
                "complexity": helpsteer_rewards_pred[3],
                "verbosity": helpsteer_rewards_pred[4],}

    def batch_inference(self, batch_messages):
        
        # batch_input_ids = self.tokenizer
        encoded_inputs = self.tokenizer(
            batch_messages,
            padding='longest',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length,
        )
        batch_size = len(batch_messages)
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = encoded_inputs[k].to(self.device)
        # import pdb; pdb.set_trace()
        batch_scores = []
        with torch.no_grad():
            output = self.model(**encoded_inputs)
            # output.rewards
            for idx in range(batch_size):
                # import pdb; pdb.set_trace()
                score = output.score[idx].float().item()
                # helpfulness, correctness, coherence, complexity, verbosity
                helpsteer_rewards_pred = (output.rewards[idx, :5] * 5 - 0.5).tolist()
                safe_pred = (output.rewards[idx, 10] * 5 - 0.5).float().item()
                score_ =    {"score": score, 
                            "helpfulness": helpsteer_rewards_pred[0],
                            "correctness": helpsteer_rewards_pred[1],
                            "coherence": helpsteer_rewards_pred[2],
                            "complexity": helpsteer_rewards_pred[3],
                            "verbosity": helpsteer_rewards_pred[4],
                            "is_safe": safe_pred}
                batch_scores.append(score_)
        # return {"score": score, "rewards": helpsteer_rewards_pred}
        
        return batch_scores
            
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--save_name', help='path where the model results to be saved', type=str, required=False, default=None)
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--response_file', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)
parser.add_argument('--exp_type', help="exp type, 'summary' or 'assistant' ", type=str, default="assistant")
parser.add_argument('--reward_type', help="", type=str, default="only_response")
parser.add_argument('--reward_names', help="", type=str, default="harmless,helpful")
parser.add_argument('--batch_size', help='batch_size', type=int, required=False, default=4)

args = parser.parse_args()

batch_size = args.batch_size
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

def process_data(dataset_name, nsamples, tokenizer = None, reward_type = "only_response"):
    f = open(dataset_name)

    dataset = json.load(f)
    new_dataset = []
    if nsamples == -1:
        nsamples = len(dataset)
    dataset = dataset[:nsamples]
    for idx, data in enumerate(dataset):
        question = data['question']
        if type(data['llm_response']) == list:
            for resp_idx, response in enumerate(data['llm_response']):
                new_data = {}
                new_data['question'] = question

                if reward_type == "only_response":
                    new_data['response'] = response
                elif reward_type == "cot_response":
                    reasoning = data['think'][resp_idx]
                    if "<think>\n" in reasoning:
                        response = "{}</think>\n\n{}".format(reasoning, response)
                    else:
                        response = "<think>\n{}</think>\n\n{}".format(reasoning, response)
                    
                    new_data['response'] = response
                message = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                chat_msg = tokenizer.apply_chat_template(
                                message,
                                tokenize = False,
                            )
                new_data['chat_msg'] = chat_msg
                new_data['ori_idx'] = idx
                new_dataset.append(new_data)
        else:
            pass
        if idx == 0:
            print("question: {}\nresponse: {}".format(question, response))
    return new_dataset, dataset


model_path = "/home/dwu/local_models/ArmoRM-Llama3-8B-v0.1"

reward_model = ArmoRMPipeline(model_path, trust_remote_code=True)

dataset, ori_dataset = process_data(dataset_name, num_samples, tokenizer=reward_model.tokenizer, reward_type = args.reward_type)

rewards_list = {}

reward_model.model.eval()

start_idx = 0
batch_times = math.ceil(len(dataset) / batch_size)
# import pdb; pdb.set_trace()
for _ in tqdm(range(batch_times)):
    if start_idx >= len(dataset):
        break
    end_idx = min(len(dataset), start_idx + batch_size)
    batch_messages = [ dataset[idx]["chat_msg"] for idx in range(start_idx, end_idx) ]
    batch_scores = reward_model.batch_inference(batch_messages)
    for idx in range(start_idx, end_idx):
        dataset[idx]['scores'] = batch_scores[idx - start_idx]
    start_idx = start_idx + batch_size

for data in dataset:
    ori_idx = data['ori_idx']
    if ori_dataset[ori_idx].get("scores", None) == None:
        ori_dataset[ori_idx]['scores'] = []
    ori_dataset[ori_idx]['scores'].append(data['scores'])

print("Reward Score:")
avg_rewards_list = []
for k, v in rewards_list.items():
    avg_rewards_list.append(v)
    print("\t{}: {}".format(k, np.mean(v)))
print("Avg Score: {}".format(np.mean(avg_rewards_list)))
# import pdb; pdb.set_trace()
outputs = []

print("evaluation reward...\n")

file = args.response_file
reward_model_name = model_path.split("/")[-1]
dataset_name = file.replace(".json",'').split('/')[-2]
file_ = dataset_name + "/" + file.replace(".json",'').split('/')[-1]
# reward_model_name = model_name.split("/")[-1]
# save_name = f"{save_path}/{file_}_{reward_model_name}_{args.reward_type}_rewards.json"
save_name = f"{save_path}/{file_}_{reward_model_name}.json"

with open(f'{save_name}', 'w', encoding='utf-8') as f:
    json.dump(ori_dataset, f, ensure_ascii=False, indent=4)

print(f"\nCompleted, pelase check {save_name}")
