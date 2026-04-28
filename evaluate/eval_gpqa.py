import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import re
import sys
from vllm import LLM, SamplingParams
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上上级目录的路径
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 添加到 sys.path
sys.path.append("/home/dwu/Immunization")
from src import system_template

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

XFINDER_SYSTEM_PROMPT = "You are a help assistant tasked with extracting the precise key answer from given output sentences."

xfinder_query_template  = 'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '

XFINDER_PROMPT_TEMPLATE = {
    "xFinder-qwen1505":
        """<|System|>:{system}
<|User|>:{input}
<|Bot|>:""",
    "xFinder-llama38it":
        """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
}

class ARCDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, system_prompt=None, use_cot_prompt = True, num_samples = -1):
        if num_samples != -1:
            data_list = data_list[:num_samples]
        
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.features = []
        self.prompt_list = []
        for idx, data in enumerate(self.data_list):
            
            instruction = '''Please choose one or more options as the answer to the following question.'''
            question = data["question"]
            options_text = ""
            for label, option in zip(data["choices"]["label"], data["choices"]["text"]):
                options_text += f"({label}): {option}\n"
            
            prompt = f"{instruction}\nQuestion: {question}\n{options_text}"
            prompt = prompt + '''\nYour response must end with format "The answer is".'''
            messages = []
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if use_cot_prompt:
                prefix = "Let's think step by step."
            else:
                prefix = "The answer is"
            messages.append({"role": "assistant", "content": prefix})

            # chat_template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = False)
            # Remove eos_token
            prompt = prompt.split(tokenizer.eos_token)[0].strip()
            self.prompt_list.append(prompt)
            data_item = self.tokenizer(prompt, padding=True, return_tensors="pt", add_special_tokens = False)
            data_item = {k: v[0] for k, v in data_item.items()}
            if idx == 0:
                pass
                decode_text = self.tokenizer.decode(data_item["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{[decode_text]}")
            self.features.append(data_item)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

def get_prompts(dataset, system_prompt = None):
    prompts = []
    new_dataset = []
    for raw_data in dataset:
        # import pdb; pdb.set_trace()
        instruction = '''Please choose one options as the answer to the following question.'''
        question = raw_data["Question"]
        options_text = ""

        data = {}
        data['question'] = question
        data['choices'] = {}
        data['choices']["label"] = ["A", "B", "C", "D"]
        data['choices']['text'] = [raw_data['Correct Answer'], raw_data['Incorrect Answer 1'], raw_data['Incorrect Answer 2'], raw_data['Incorrect Answer 3']]
        data['answerKey'] = "A"
        # import pdb; pdb.set_trace()

        for label, option in zip(data["choices"]["label"], data["choices"]["text"]):
                options_text += f"({label}): {option}\n"
        prompt = f"{instruction}\nQuestion: {question}\n{options_text}"
        prompt = prompt + '''\nYour response must end with format "The answer is".'''
        messages = []
        if system_prompt != None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
        # import pdb; pdb.set_trace()
        prompts.append(prompt)
        new_dataset.append(data)
    return prompts, new_dataset

def extract_answer(text, option_list):
    # pass
    # regex_pattern = r'(?i)the answer is\s*(.*)'
    regex_pattern = r'(?:the answer is|The answer is)\s*(.*?)(?=\s*\.|$)'

    # 使用re.findall找到所有匹配的内容
    matches = re.findall(regex_pattern, text, re.IGNORECASE)

    # 如果有匹配的内容，返回最后一个匹配项
    if matches:
        sub_text = matches[-1].strip()
    else:
        sub_text = text
    
    # sub_text = text.split("The answer is")[-1]
    # regex_pattern = r'[A-D](?=[\s:.])'
    regex_pattern = r'(' + '|'.join(re.escape(option) for option in option_list) + r')(?=[\s:.()\n\t]|$)'

    matches = re.findall(regex_pattern, sub_text)
    # import pdb; pdb.set_trace()
    # print(matches)
    preds = list(set(matches))
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
    parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
    parser.add_argument('--save_name', help='path where the model results to be saved', type=str, required=False, default=None)
    parser.add_argument('--tag', help='', type=str, required=False, default="")
    parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
    parser.add_argument('--tensor_parallel_size', help='tensor_parallel_size', type=int, required=False, default=1)
    parser.add_argument('--subset', help='', choices=["main", "diamond", "Both"])
    parser.add_argument('--need_system_prompt', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=False, type=int, default=0)
    parser.add_argument('--max_new_tokens', help='max_new_tokens', required=False, type=int, default=8192)
    parser.add_argument('--temperature', help='temperature', required=False, type=float, default=0.00)
    parser.add_argument('--top_p', help='top_p', required=False, type=float, default=1.00)
    parser.add_argument('--seed', help='seed', required=False, type=int, default=42)
    parser.add_argument('--n_generation', help='n_generation', required=False, type=int, default=1)
    
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    subset = args.subset
    model_name = args.model
    save_path = args.save_path
    num_samples = args.num_samples

    if args.need_system_prompt == 0:
        system_prompt = None
    else:
        pass
        
    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    ##process data
    dataset_dict = {}
    if subset == "main":
        file_path = "/home/dwu/Immunization/evaluate/gpqa_main.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict[subset] = data_list[:args.num_samples]
    elif subset == "diamond":
        file_path = "/home/dwu/Immunization/evaluate/gpqa_diamond.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict[subset] = data_list[:args.num_samples]

    gpu_memory_utilization = 0.85
    stop_words = [tokenizer.eos_token]
    
    llm = LLM(model_name,
        max_model_len = args.max_new_tokens + 4096,
        tensor_parallel_size = args.tensor_parallel_size,
        gpu_memory_utilization = gpu_memory_utilization)

    print("generating responses...\n")
    # import pdb; pdb.set_trace()
    dataset_result_dict = {}
    over_write = False
    for dataset_name, data_list in dataset_dict.items():

        if args.save_name != None:
            save_name = f"{save_path}/{args.save_name}"
        else:
            save_name = f'{save_path}/gpqa_{subset}/{model_name.split("/")[-1]}{args.tag}_sys_{args.need_system_prompt}'
        
        if not os.path.exists(save_name):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(save_name)
        # elif over_write == False:
        # else:

        prompts, data_list = get_prompts(dataset = data_list)
        print("----------------------------------------------------")
        print("input:\n{}".format(prompts[0]))
        print("----------------------------------------------------")

        llm_outputs = llm.generate(prompts, SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        # use_beam_search=use_beam_search,
                        max_tokens=args.max_new_tokens,
                        n=args.n_generation,
                        stop=stop_words,
                        seed=args.seed,
        ))
        
        llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        
        outputs_list = []
        for output in llm_outputs:
            outputs_list.append([output.outputs[idx].text for idx in range(len(output.outputs))])
        # import pdb; pdb.set_trace()
        max_score = len(prompts) * 1.0
        cumulative_score = 0.0
        dataset_result_dict[dataset_name] = {}
        result_record = []
        dataset_result_dict[dataset_name]["max_score"] = max_score

        #####
        del llm

        xfinder_model_name = "xFinder-qwen1505"
        key_answer_type = "alphabet option"
        xfinder_inputs = []
        for idx in  range(len(data_list)) :
            question = data_list[idx]['question']
            # import pdb; pdb.set_trace()
            llm_output = outputs_list[idx][0].split("</think>")[-1]
            standard_answer_range = []
            for label, option in zip(data_list[idx]["choices"]["label"], data_list[idx]["choices"]["text"]):
                    # options_text += f"({label}): {option}\n"
                    standard_answer_range.append([label, option])
            extract_formatted_query = xfinder_query_template.format(question = question, llm_output = llm_output, standard_answer_range=str(standard_answer_range))
            xfinder_prompt = XFINDER_PROMPT_TEMPLATE[xfinder_model_name].format(system=XFINDER_SYSTEM_PROMPT, input=extract_formatted_query)
            xfinder_inputs.append(xfinder_prompt)

        model_name_or_path = "/home/dwu/local_models/{}".format(xfinder_model_name)
        xfinder = LLM(model_name_or_path, tensor_parallel_size = args.tensor_parallel_size)
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/dwu/local_models/{}".format(xfinder_model_name), trust_remote_code=True
        )
        stop_words = []
        stop_words.append(tokenizer.eos_token)
        stop_words.append("<|eot_id|>")
        xfinder_outputs = xfinder.generate(xfinder_inputs, SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=32,
                    n=1,
                    stop=stop_words,
        ))
        xfinder_outputs = sorted(xfinder_outputs, key=lambda x: int(x.request_id))
        xfinder_outputs_list = [output.outputs[0].text for output in xfinder_outputs]

        
        # import pdb; pdb.set_trace()
        for idx, responses in enumerate(outputs_list):

            # response = tokenizer.decode(response_ids, skip_special_tokens=True)
            # import pdb; pdb.set_trace()

            # response_text_list.append(response)
            response = responses[0]
            # answer = response.split("</think>")[-1]
            answer = xfinder_outputs_list[idx]
            preds = extract_answer(answer, data_list[idx]['choices']['label'])

            correct_answer = data_list[idx]['answerKey']
            # import pdb; pdb.set_trace()
            # question = orig_que[idx]
            question = data_list[idx]['question']

            num_answers = len(preds)
            if correct_answer in preds:
                # 如果正确答案在预测中，得分是 1/N
                score = 1.0 / num_answers
            else:
                # 如果正确答案不在预测中，得分为 0
                score = 0.0
            cumulative_score += score
            
            data_result = data_list[idx]
            data_result["prompt"] = prompts[idx]
            data_result["response"] = response
            data_result["answer"] = preds
            data_result["score"] = score
            # response = [{'prompt':question, 'response':response, ""}]

            result_record.append(data_result)

        score_record = {"max_score": max_score, "cumulative_score": cumulative_score, "acc": cumulative_score / max_score * 100.0}

        print(f"\n\n[Total score]: \n{json.dumps(score_record, indent=4)}")

        save_preds_name = save_name + f"/{dataset_name}_predications.json"
        with open(f'{save_preds_name}', 'w', encoding='utf-8') as f:
            json.dump(result_record, f, ensure_ascii=False, indent=4)

        save_score_name = save_name + f"/{dataset_name}_score.json"
        with open(save_score_name, 'w') as json_file:
            json.dump(score_record, json_file, indent=4)
        print(f"\nCompleted ARC Subset {dataset_name}, pelase check {save_name}")
