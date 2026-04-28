import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams


import sys
sys.path.append("/home/dwu/")
from human_eval.data import read_problems, write_jsonl

sys.path.append("/home/dwu/ReAlign")
from src import system_template

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

Is_Deliberative_Alignment = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--lora_path', help='', type=str, required=False, default=None)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
# parser.add_argument('--dataset', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)
parser.add_argument('--max_new_tokens', help='max_new_tokens', required=False, type=int, default=512)
parser.add_argument('--use_system_prompt', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', action="store_true")
parser.add_argument('--num_samples_per_task', help='num_samples_per_task', required=False, type=int, default=1)
parser.add_argument('--use_cot_prompt', help='CoT', action="store_true")
parser.add_argument('--use_chat_template', help='Chat Template', action="store_true")

args = parser.parse_args()

# batch_size = args.batch_size
# dataset = args.dataset
model_name = args.model
save_path = args.save_path
num_samples = args.num_samples

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


##setting up model##


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = LLM(model_name)


def gen_prompt(que):
    chat = [{"role": "user", "content": que}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt

def process_data(dataset, nsamples):
    f = open(dataset)

    data = json.load(f)

    if 'harmfulq' in dataset or 'cat' in dataset:
        topics = []
        subtopics = []
        prompt_que = []
        orig_que = []
        for topic in data.keys():
            for subtopic in data[topic].keys():
                for q in data[topic][subtopic]:
                    orig_que.append(q)
                    prompt_que.append(gen_prompt(q))
                    topics.append(topic)
                    subtopics.append(subtopic)

    else:
        prompt_que = [gen_prompt(q) for q in data]
        orig_que = data
        topics, subtopics = [], []

    if nsamples == -1:
        nsamples = len(prompt_que)

    return prompt_que[:nsamples], orig_que[:nsamples], topics[:nsamples], subtopics[:nsamples]
dataset = "/home/dwu/human-eval/data/HumanEval.jsonl.gz"
problems = read_problems(dataset)
# import pdb; pdb.set_trace()
# prompt_que, orig_que, topics, subtopics = process_data(dataset, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if args.lora_path != None:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".jsonl.gz","")}/{model_name.split("/")[-1]}_{args.lora_path.split("/")[-1]}_sys_{args.use_system_prompt}_@{args.num_samples_per_task}_chat_{args.use_chat_template}.jsonl'
else:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".jsonl.gz","")}/{model_name.split("/")[-1]}_sys_{args.use_system_prompt}_@{args.num_samples_per_task}_chat_{args.use_chat_template}.jsonl'
outputs = []
system_message = ''


print("generating responses...\n")


class HumanEvalDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, system_prompt=None, num_samples_per_task = 16, use_cot_prompt = False, use_chat_template = False, num_samples = -1):
        
        # if num_samples != -1:
        #     data_list = data_list[:num_samples]
        self.data_list = []
        for _, data in data_list.items():
            # task_id = data['task_id']

            # prompt = data['prompt']
            # data = data['prompt'].replace("    ", "\t")
            # data = data['prompt']
            self.data_list.append({"task_id": data['task_id'], "prompt": data["prompt"]})
        if num_samples != -1:
            self.data_list = self.data_list[:num_samples]   

        # self.data_list = data_list
        self.tokenizer = tokenizer
        self.features = []
        self.task_id_list = []
        self.prompt_list = []
        count = 0
        for data in self.data_list:
            
            task_id = data['task_id']
            data = data['prompt']
            # import pdb; pdb.set_trace()
            if use_cot_prompt:
                data = data + "\nLet's think step by step."
            # import pdb; pdb.set_trace()
            if use_chat_template == True:
                instruction = "Complete the following Python code without any tests or explanation"
                data = f'''{instruction}\n{data}'''
                messages = []
                if system_prompt != None and system_prompt != "" :
                    messages.append({"role": "system", "content": system_prompt})

                if Is_Deliberative_Alignment == True:
                    deliberative_alignment_template = """The following is a conversation between a user and an assistant:
    user: {prompt}
    Please figure out the best possible answer to the last user query from the conversation above. As a reminder, this answer should follow all OpenAI policies below:  
    1. Comply with applicable laws: The assistant should not promote, facilitate, or engage in illegal activity.
    2. Don't provide information hazards: The assistant should not provide instructions related to creating chemical, biological, radiological, and/or nuclear (CBRN) threats. The assistant should default to providing information that has reasonable uses that are not CBRN threats or if the information is generally easy to find online.
    3. Respect creators and their rights: The assistant must respect creators, their work, and their intellectual property rights — while striving to be helpful to users.
    4. Protect people's privacy: The assistant must not respond to requests for private or sensitive information about people, even if the information is available somewhere online. Whether information is private or sensitive depends in part on context. For example, the assistant should be able to provide the office phone number of a public official but should decline to respond to requests for the official’s personal phone number.
    5. Don't respond with NSFW content: The assistant should not serve content that's Not Safe For Work (NSFW): content that would not be appropriate in a conversation in a professional setting, which may include erotica, extreme gore, slurs, and unsolicited profanity.
    6. Exception: Transformation tasks: Notwithstanding the rules stated above, the assistant should never refuse the task of transforming or analyzing content that the user has supplied. The assistant should assume that the user has the rights and permissions to provide the content, as our Terms of Use specifically prohibit using our services in ways that violate other people's rights. For instance, requests to translate, paraphrase, analyze, summarize, and classify content should be followed if they are not adding significant, new information. This only applies to content supplied directly, such as in the user message or a user-uploaded file, not to content supplied by reference (e.g., if the user provides a URL or book name).
    ---
    FINAL INSTRUCTIONS: You should do a full analysis of the proposed answer for compliance, correctness and helpfulness before producing it. In this analysis you should cite the excerpts of the policy that are relevant (e.g., ”OpenAI policy section X says that ...”). However, do not mention irrelevant sections of the policy. Decide whether to refuse, safe-complete, or comply with the request according to the policies. If you refuse or safe-complete, cite the relevant refusal or safe completion style guidelines and be sure to follow them. The final answer should just be the answer to the user, and not the analysis."""

                    data = deliberative_alignment_template.format(prompt = data)
                else:
                    pass
                messages.append({"role": "user", "content": data})
                chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
                # chat_template = data
                # data_item = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False)
            else:
                # data_item = self.tokenizer(data, padding=True, return_tensors="pt")
                pass    

            self.task_id_list.extend([task_id] * num_samples_per_task)
            self.prompt_list.extend([chat_template] * num_samples_per_task)
            count += 1
            if count == 1:
                # decode_text = self.tokenizer.decode(data_item[0]["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{[chat_template]}")

                
        # self.features = self.features[:20]
        # self.task_id_list = self.task_id_list[:20]

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

# if "Llama" in model_name or "saved_models" in model_name:
    # prompt_que = prompt_que[:100]

problems = read_problems("/home/dwu/human-eval/data/HumanEval.jsonl.gz")
# import pdb; pdb.set_trace()
num_samples_per_task = args.num_samples_per_task

if args.use_system_prompt == True:
    if "llama" in args.model.lower():
        system_prompt = system_template.llama_2_system_prompt
else:
    system_prompt = None

human_eval_dataset = HumanEvalDataset(problems, tokenizer, num_samples_per_task = num_samples_per_task, system_prompt=system_prompt, use_cot_prompt = args.use_cot_prompt, use_chat_template = args.use_chat_template, num_samples = num_samples)
# data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
# human_eval_dataloader = DataLoader(human_eval_dataset, batch_size = batch_size, collate_fn=data_collator)

outputs_list = []
response_text_list = []
generation_util = [
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|start_header_id|>",
]
generation_util.append(tokenizer.eos_token)
prompts = human_eval_dataset.prompt_list
if num_samples_per_task == 1:
    temperature = 0.0
else:
    temperature = 0.2
top_p = 1.0
max_tokens = args.max_new_tokens
outputs = model.generate(prompts, SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
                stop=generation_util,
))
outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
outputs_list = [output.outputs[0].text for output in outputs]

outputs = []
samples = []

def filter_code_v1(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def filter_code_v2(completion: str) -> str:
    # completion = completion.lstrip("\n")
    # return completion.split("\n\n")[0]
    completion = completion.lstrip("\n")
    import re
    # import pdb; pdb.set_trace()

    code_blocks = re.findall(r'```(.*?)```', completion, re.DOTALL)
    if len(code_blocks) == 0:
        code_blocks = completion.split("\n")
        mark_idx = 0
        for idx, line in enumerate(code_blocks):
            if "def" in line:
                mark_idx = idx
        new_code = "\n".join(code_blocks[mark_idx + 1:])
    else:
        code_blocks = code_blocks[0]
        code_blocks = code_blocks.split("\n")
        mark_idx = 0
        for idx, line in enumerate(code_blocks):
            if "def" in line:
                mark_idx = idx
        new_code = "\n".join(code_blocks[mark_idx + 1:])
    return new_code

for idx, response in enumerate(outputs_list):
    # import pdb; pdb.set_trace()
    # response = tokenizer.decode(response_ids, skip_special_tokens=True)
    if "DeepSeek-R1" in model_name or "</think>" in response:
        think_cot = response.split("</think>")[0]
        response = response.split("</think>")[-1]
    # response = response.replace("\t", "    ")
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    if args.use_chat_template == True:
        # response = filter_code_v1(response)
        response = filter_code_v2(response)
    else:
        response = filter_code_v1(response)
    # import pdb; pdb.set_trace()
    new_data = dict(task_id = human_eval_dataset.task_id_list[idx], prompt=human_eval_dataset.prompt_list[idx], completion=response,)
    if "DeepSeek-R1" in model_name or "</think>" in response:
        new_data['think'] = think_cot
    samples.append(new_data)


from typing import Iterable, Dict
import gzip

write_jsonl(save_name, samples)
print(f"\nCompleted, please check {save_name}")
