import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上上级目录的路径
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 添加到 sys.path
sys.path.append("/home/dwu/ReAlign")
from src import system_template

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM




parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--lora_path', help='', type=str, required=False, default=None)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--save_name', help='path where the model results to be saved', type=str, required=False, default=None)
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--dataset', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)
parser.add_argument('--need_system_prompt', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=False, type=int, default=0)
parser.add_argument('--batch_size', help='batch_size', required=False, type=int, default=8)


args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset
model_name = args.model
save_path = args.save_path
num_samples = args.num_samples

if args.need_system_prompt == 0:
    system_prompt = None
else:
    if "open-instruct-llama2-sharegpt-dpo-7b" in args.model or "tulu" in args.model:
        system_prompt = None
    elif "llama" in args.model.lower():
        # system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
        # system_prompt = 
        system_prompt = system_template.llama_2_system_prompt
    elif "mistral" in args.model.lower():
        system_prompt = None
    elif "internlm2" in args.model.lower():
        system_prompt = system_template.internlm_2_system_prompt
    else:
        system_prompt = system_template.llama_2_system_prompt
    


print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


##setting up model##
# if 'gpt' in model_name and len(model_name) < 10:

#     import openai

#     try:
#         # API setting constants
#         API_MAX_RETRY = 5
#         API_RETRY_SLEEP = 10
#         API_ERROR_OUTPUT = "$ERROR$"

#         key_path = f'api_keys/{model_name}_api_key.json'
#         with open(key_path, 'r') as f:
#             keys = json.load(f)   

#         openai.api_type = keys['api_type']
#         openai.api_base = keys['api_base']
#         openai.api_version = keys['api_version']
#         openai.api_key=keys['api_key']
#         model_engine = keys['model_engine']
#         model_family = keys['model_family']
#     except:
#         raise Exception(f"\n\n\t\t\t[Sorry, please verify API key provided for {model_name} at {key_path}]")

# elif 'claude' in model_name:

#     from anthropic import Anthropic

#     try:
#         # API setting constants
#         API_MAX_RETRY = 5
#         API_RETRY_SLEEP = 10
#         API_ERROR_OUTPUT = "$ERROR$"

#         key_path = f'api_keys/{model_name}_api_key.json'
#         with open(key_path, 'r') as f:
#             keys = json.load(f)   

#         anthropic = Anthropic(api_key=keys['api_key'])

#     except:
#         raise Exception(f"\n\n\t\t\t[Sorry, please verify API key provided for {model_name} at {key_path}]")



# else:

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)

if "open-instruct-llama2-sharegpt-dpo-7b" in args.model or "tulu" in args.model:
    llama_tokenizer = AutoTokenizer.from_pretrained("/home/dwu/local_models/open-instruct-llama2-sharegpt-dpo-7b", padding_side="left", use_fast=False)
    # tokenizer.chat_template = llama_tokenizer.chat_template
    tokenizer.chat_template = '''{% for message in messages %}
{% if message['role'] == 'user' %}
    {{ bos_token + '<|user|>\n' + message['content'].strip() + '\n<|assistant|>'}}
{% elif message['role'] == 'assistant' %}
    {{ message['content'] + eos_token }}
{% endif %}
{% endfor %}
'''
elif "llama" in args.model.lower():
    pass
    # llama_tokenizer = AutoTokenizer.from_pretrained("/home/dwu/local_models/Llama-2-7b-chat-hf", padding_side="left", use_fast=False)
    # tokenizer.chat_template = llama_tokenizer.chat_template
elif "mistral" in args.model.lower():
    mistral_tokenizer = AutoTokenizer.from_pretrained("/home/dwu/local_models/Mistral-7B-Instruct-v0.2", padding_side="left", use_fast=False)
    # tokenizer.chat_template = mistral_tokenizer.chat_template
# elif ""
tokenizer.pad_token = tokenizer.eos_token

if "4bit" in model_name:
    # torch_dtype = torch.bfloat16
    torch_dtype = None
    load_in_4bit = True
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype,
    load_in_4bit = load_in_4bit,
    )
else:

    torch_dtype = torch.bfloat16
    # load_in_4bit = True
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype,
    # load_in_4bit = load_in_4bit,
    )

if args.lora_path != None:
    adapter_model_id  = args.lora_path
    print("Load Lora from:{}".format(adapter_model_id))
    model = PeftModel.from_pretrained(model, adapter_model_id)
for name, param in model.named_parameters():
    if name.find("lora_") != -1:
        # param
        # import pdb; pdb.set_trace()
        layer_num = int(name.split("base_model.model.model.layers.")[-1].split(".")[0])
        if layer_num > 25:
            # import pdb; pdb.set_trace()
            param.data = torch.zeros_like(param.data)
                # param.requires_grad = False
##define chat completion function for GPT##
# def chat_completion_gpt(system, prompt):
#     for _ in range(API_MAX_RETRY):
#         try:    
#             response = openai.ChatCompletion.create(
#               model=model_family,
#               engine=model_engine,
#               messages=[
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": prompt},
#                 ]
#             )
#             response = response['choices'][0]['message']['content']
#             return response
#         except Exception as e:
#             if 'policy' in str(e):
#                 print("Skipping due to openai policy")
#                 #such cases are tagged with [[N]]
#                 return '[[N]]'
#             print(type(e), e)
#             print("trying again")
#             time.sleep(API_RETRY_SLEEP)


##define chat completion function for Claude##
# def chat_completion_claude(system, prompt):

#     try_suffix = [" Let's think step by step:"," Let's think step by step.", " Let's think step by step:\n-", " Let's think step by step.\n-", ""]

#     for t in range(API_MAX_RETRY):
#         try:
#             full_prompt = f"{prompt}{try_suffix[t]}"
#             print(f"Retry:{t}\n{full_prompt}")
#             completion = anthropic.completions.create(
#                 model=model_name,
#                 max_tokens_to_sample=300,
#                 prompt=full_prompt,
#             )
#             response = completion.completion
#             if response == "" or "Human" in response:
#                 continue
                
#             return response

#         except Exception as e:
#             print(type(e), e)
#             print("trying again")
#             time.sleep(API_RETRY_SLEEP)

#             return response

##process data
def clean_thoughts_(response):

    if "(Internal thought:" in response:
        if ')' in response:
            ind =  response.index(')')+1
        else:
            ind = -1
        nresponse = response[ind:].strip()
        return nresponse

    return response


def get_context(file_name):
    f = open(file_name, "r")
    f = f.read()
    return f

def gen_prompt(que, system_prompt = None):
    chat = []
    if system_prompt != None and system_prompt != "" :
        chat.append({"role": "system", "content": system_prompt})
    # que = "Construct a function call to answer the given question using keyword arguments based on the provided specifications:\n\nQuestion: " + que
    chat.append({"role": "user", "content": que})
    # import pdb; pdb.set_trace()
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt = True)
    # prompt = prompt + " I"
    # import pdb; pdb.set_trace()
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
                    prompt_que.append(gen_prompt(q, system_prompt = system_prompt))
                    topics.append(topic)
                    subtopics.append(subtopic)
    elif "salad" in dataset.lower():
        if "attack_enhanced_set" in dataset.lower():
            data_key = "prompt"
        elif "base_set" in dataset.lower():
            data_key = "prompt"
        prompt_que = [gen_prompt(q[data_key], system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    elif "openfunction" in dataset.lower():
        # import pdb; pdb.set_trace()
        prompt_que = [gen_prompt(q["instruction"], system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    elif "wild_jailbreak" in dataset.lower():
        # import pdb; pdb.set_trace()
        prompt_que = [gen_prompt(q["prompt"], system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    elif type(data[0]) == dict and data[0].get("instruction", None) != None and data[0].get("input", None) != None:
        # prompt = ""

        prompt_que = [gen_prompt('''{}\n{}'''.format(q['instruction'] if q.get("instruction", None) != None else "", q['input'] if q.get("input", None) != None else ""), system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    else:
        prompt_que = [gen_prompt(q, system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []

    if nsamples == -1:
        nsamples = len(prompt_que)

    return prompt_que[:nsamples], orig_que[:nsamples], topics[:nsamples], subtopics[:nsamples]


prompt_que, orig_que, topics, subtopics = process_data(dataset, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if args.save_name != None:
    save_name = f"{save_path}/{args.save_name}"
elif args.lora_path != None:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}/{model_name.split("/")[-1]}_{args.lora_path.split("/")[-1]}_sys_{args.need_system_prompt}.json'
else:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}/{model_name.split("/")[-1]}_sys_{args.need_system_prompt}.json'
outputs = []
# system_message = ''



print("generating responses...\n")


class TmpDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, system_prompt=None):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.features = []
        truncation_count = 0
        for idx, data in enumerate(self.data_list):
            # import pdb; pdb.set_trace()
            # messages = []
            # if system_prompt != None and system_prompt != "" :
            #     messages = [{"role": "system", "content": system_prompt}]
                # system_message +=
                # pass
            # else:
            # messages.append({"role": "user", "content": data})

            # chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
            # import pdb; pdb.set_trace()
            chat_template = data
            # import pdb; pdb.set_trace()
            data_item = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False, truncation=True, max_length = 1500)
            # data_b = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False, truncation=True)
            if data_item['input_ids'].shape[-1] < data_item["input_ids"].shape[-1]:
                truncation_count += 1

            data_item = {k: v[0] for k, v in data_item.items()}
            if idx == 0:
                pass
                decode_text = self.tokenizer.decode(data_item["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{[decode_text]}")
            
            self.features.append(data_item)
        print(f"truncation_count: {truncation_count}")

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

if "llama" in model_name.lower() or "saved_models" in model_name.lower() or ("gpt" not in model_name.lower() and "claude" not in model_name.lower()):
    # prompt_que = prompt_que[:100]
    

    tmp_dataset = TmpDataset(prompt_que, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
    tmp_dataloader = DataLoader(tmp_dataset, batch_size = batch_size, collate_fn=data_collator)
    class StopAtSpecificTokenCriteria(StoppingCriteria):
        def __init__(self, token_id_list, batch_size):
            # import pdb; pdb.set_trace()
            self.token_id_list = token_id_list
            self.count_eos_batch = [False for _ in range(batch_size)]
        
        def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
            # import pdb; pdb.set_trace()
            result = True
            for i, tmp_idx in enumerate(input_ids):
                if tmp_idx[-1].detach().cpu().numpy() in self.token_id_list:
                    self.count_eos_batch[i] = True
            for eos in self.count_eos_batch:
                result = result and eos
            return result
    outputs_list = []
    response_text_list = []
    model.eval()
    if "fft" in  model_name:
        max_new_tokens = 512
    else:
        max_new_tokens = 512
    for batch in tqdm(tmp_dataloader, total=len(tmp_dataloader)):
        stopping_criteria = []
        if "llama-3" in model_name.lower() or "llama3" in model_name.lower():
            # stop_ids.append(128009)
            # stop_criteria = StoppingCriteria(stop_ids)
            tmp_batch_size = batch['input_ids'].shape[0]
            # stopping_criteria.append(StoppingCriteria([tokenizer.convert_tokens_to_ids("<|eot_id|>")], batch_size))
            stopping_criteria = StoppingCriteriaList()
            # stopping_criteria.append(StopAtSpecificTokenCriteria(tokenizer.eos_token_id], tmp_batch_size))
            stopping_criteria.append(StopAtSpecificTokenCriteria([tokenizer.convert_tokens_to_ids("<|eot_id|>")], tmp_batch_size))
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            if len(stopping_criteria) == 0:
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    # temperature = 0,
                    # do_sample=False,
                    use_cache=True,
                )
            else:
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria = stopping_criteria,
                    # temperature = 0,
                    # do_sample=False,
                    use_cache=True,
                )
            seq_len = batch['input_ids'].shape[-1]
            outputs = outputs[:, seq_len:]
            outputs_list.extend(outputs.tolist())
    outputs = []
    for idx, response_ids in enumerate(outputs_list):
        if "llama3" in model_name.lower() or "llama-3" in model_name.lower():
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            # import pdb; pdb.set_trace()
            try:
                index = response_ids.index(eot_id)
                response_ids = response_ids[ : index + 1]
            except:
                pass
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_text_list.append(response)

        question = orig_que[idx]
        question2 = prompt_que[idx].replace('<s>','')
        
        #cleaning response
        response = response.replace(question2,"").strip()

        if 'zephyr' in model_name:
            response = response[response.index(question2[-10:]):][10:]

        if 'harmfulq' in dataset or 'cat' in dataset:
            response = [{'prompt':question, 'response':response, 'topic':topics[idx], 'subtopic': subtopics[idx]}]
        else:
            response = [{'prompt':question, 'response':response}]

        outputs += response

        with open(f'{save_name}', 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)


else:

    for i in tqdm(range(len(prompt_que))):

        inputs = prompt_que[i]

        if 'gpt' in model_name and len(model_name) < 10:
            response = chat_completion_gpt(system=system_message, prompt=inputs)

        elif 'claude' in model_name:
            response = chat_completion_claude(system=system_message, prompt=inputs)

        else:
            inputs = tokenizer([inputs], return_tensors="pt", truncation=True, padding=True).to("cuda")
            generated_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=500)
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        question = orig_que[i]
        question2 = prompt_que[i].replace('<s>','')
        
        #cleaning response
        response = response.replace(question2,"").strip()

        if 'zephyr' in model_name:
            response = response[response.index(question2[-10:]):][10:]

        if 'harmfulq' in dataset or 'cat' in dataset:
            response = [{'prompt':question, 'response':response, 'topic':topics[i], 'subtopic': subtopics[i]}]
        else:
            response = [{'prompt':question, 'response':response}]

        outputs += response

        with open(f'{save_name}', 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)

print(f"\nCompleted, pelase check {save_name}")
