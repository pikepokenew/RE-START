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
self_align_v2_template = '''The following is a conversation between a user and an assistant:
user: ###{question}###
Please figure out the best possible answer to the user query. As a reminder, this answer should follow all OpenAI Policies below:
1. Comply with laws and ethics: The assistant must not promote, assist with, or engage in illegal or unethical activities. It should proactively refuse requests involving illegal, immoral, or harmful topics, prioritizing user safety, ethical conduct, and responsible principles in all responses.
2. Don't provide information hazards: The assistant should not provide instructions related to creating chemical, biological, radiological, and/or nuclear (CBRN) threats. The assistant should default to providing information that has reasonable uses that are not CBRN threats or if the information is generally easy to find online.
3. Respect creators and their rights: The assistant must respect creators, their work, and their intellectual property rights — while striving to be helpful to users.
4. Protect people's privacy: The assistant must not respond to requests for private or sensitive information about people, even if the information is available somewhere online. Whether information is private or sensitive depends in part on context. For example, the assistant should be able to provide the office phone number of a public official but should decline to respond to requests for the official’s personal phone number.
5. Don't respond with NSFW content: The assistant should not serve content that's Not Safe For Work (NSFW): content that would not be appropriate in a conversation in a professional setting, which may include erotica, extreme gore, slurs, and unsolicited profanity.
On the other hand, this answer should follow all General Rules below:
1. Informative: The assistant should provide users with accurate, relevant, and up-to-date information in its responses, ensuring that the content is both educational and engaging.
2. Helpful: The assistant's responses should be positive, interesting, helpful and engaging.
3. Reasoning: The assistant's logic and reasoning should be rigorous, intelligent and defensible.
4. Multi-aspect: The assistant can provide additional relevant details to respond thoroughly and comprehensively, so as to cover multiple aspects in depth.
5. Balanced & Informative Perspectives: In discussing controversial topics, The assistant should fairly and impartially present extensive arguments from both sides.
---
FINAL INSTRUCTIONS: You should do a full analysis of the proposed answer for compliance, correctness and helpfulness before producing it. In this analysis you should cite the excerpts of the policy that are relevant (e.g., OpenAI policy section X says that ...”). However, do not mention irrelevant sections of the policy. Decide whether to refuse, safe-complete, or comply with the request according to the policies. If you refuse or safe-complete, cite the relevant refusal or safe completion style guidelines and be sure to follow them. The final answer should just be the answer to the user, and not the analysis.'''
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
    for data in dataset:
        # import pdb; pdb.set_trace()
        instruction = '''Please choose one or more options as the answer to the following question.'''
        question = data["question"]
        options_text = ""
        for label, option in zip(data["choices"]["label"], data["choices"]["text"]):
                options_text += f"({label}): {option}\n"
        prompt = f"{instruction}\nQuestion: {question}\n{options_text}"
        prompt = prompt + '''\nYour response must end with format "The answer is".'''
        messages = []
        if system_prompt != None:
            # messages.append({"role": "system", "content": system_prompt})
            prompt = system_prompt.format(question = prompt)
        messages.append({"role": "user", "content": prompt})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
        # import pdb; pdb.set_trace()
        prompts.append(prompt)
    return prompts

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
    parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
    parser.add_argument('--tensor_parallel_size', help='tensor_parallel_size', type=int, required=False, default=1)
    parser.add_argument('--subset', help='', choices=["Challenge", "Easy", "Both"])
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
    elif args.need_system_prompt == 14:
        system_prompt = self_align_v2_template
    else:
        system_prompt = 0
        
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
    if subset == "Challenge":
        file_path = "/home/dwu/ReAlign/evaluate/general/ai2_arc_ARC-Challenge.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict[subset] = data_list[:args.num_samples]
    elif subset == "Easy":
        file_path = "/home/dwu/ReAlign/evaluate/general/ai2_arc_ARC-Easy.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict[subset] = data_list[:args.num_samples]
    elif subset == "Both":
        file_path = "/home/dwu/ReAlign/evaluate/general/ai2_arc_ARC-Easy.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict["ARC-Easy"] = data_list[:args.num_samples]

        file_path = "/home/dwu/ReAlign/evaluate/general/ai2_arc_ARC-Challenge.json"
        data_list = json.load(open(file_path, 'r'))
        dataset_dict["ARC-Challenge"] = data_list[:args.num_samples]
    gpu_memory_utilization = 0.8
    stop_words = [tokenizer.eos_token]
    # stop_words.append(tokenizer.eos_token)
    if "llama3" in model_name.lower() or "llama-3" in model_name.lower():
        stop_words.append("<|eot_id|>")
        stop_words.append("<|start_header_id|>")
    if "gemma" in model_name.lower():
        stop_words.append("<end_of_turn>")
    llm = LLM(model_name,
        max_model_len = args.max_new_tokens + 4096,
        tensor_parallel_size = args.tensor_parallel_size,
        gpu_memory_utilization = gpu_memory_utilization)

    print("generating responses...\n")
    # import pdb; pdb.set_trace()
    dataset_result_dict = {}
    llm_result = {}
    for dataset_name, data_list in dataset_dict.items():
        prompts = get_prompts(dataset = data_list, system_prompt = system_prompt)
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

        for idx in range(len(data_list)):
            data_list[idx]['response'] = outputs_list[idx][0]
        # import pdb; pdb.set_trace()
        llm_result[dataset_name] = data_list
    # 开始抽取答案
    del llm
    xfinder_model_name = "xFinder-qwen1505"
    key_answer_type = "alphabet option"
    model_name_or_path = "/home/dwu/local_models/{}".format(xfinder_model_name)
    xfinder = LLM(model_name_or_path, tensor_parallel_size = args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/dwu/local_models/{}".format(xfinder_model_name), trust_remote_code=True
    )
    stop_words = []
    stop_words.append(tokenizer.eos_token)
    stop_words.append("<|eot_id|>")

    for dataset_name, data_list in llm_result.items():
        xfinder_inputs = []
        if args.save_name != None:
            save_name = f"{save_path}/{args.save_name}"
        else:
            save_name = f'{save_path}/ai2_arc/{model_name.split("/")[-1]}_sys_{args.need_system_prompt}'
        
        if not os.path.exists(save_name):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(save_name)
        for idx in range(len(data_list)):
            # import pdb; pdb.set_trace()
            question = data_list[idx]['question']
            input_prompt = question
            standard_answer_range = []
            for option, text in zip(data_list[idx]['choices']['label'], data_list[idx]['choices']['text']):
                input_prompt = input_prompt + f"\n({option}):{text}"
                standard_answer_range.append([option, text])
            # import pdb; pdb.set_trace()
            llm_output = data_list[idx]['response'].split("</think>")[-1].strip("\n ")

            extract_formatted_query = xfinder_query_template.format(question = question, llm_output = llm_output, standard_answer_range=str(standard_answer_range))
            xfinder_prompt = XFINDER_PROMPT_TEMPLATE[xfinder_model_name].format(system=XFINDER_SYSTEM_PROMPT, input=extract_formatted_query)
            xfinder_inputs.append(xfinder_prompt)


        xfinder_outputs = xfinder.generate(xfinder_inputs, SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=32,
                    n=1,
                    stop=stop_words,
        ))
        xfinder_outputs = sorted(xfinder_outputs, key=lambda x: int(x.request_id))
        xfinder_outputs_list = [output.outputs[0].text for output in xfinder_outputs]

        cumulative_score = 0
        # import pdb; pdb.set_trace()
        max_score = len(data_list)
        for idx, responses in enumerate(xfinder_outputs_list):
            # import pdb; pdb.set_trace()

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
            
            data_list[idx]['score'] = score
            # response = [{'prompt':question, 'response':response, ""}]

            # result_record.append(data_result)

        score_record = {"max_score": max_score, "cumulative_score": cumulative_score, "acc": cumulative_score / max_score * 100.0}

        print(f"\n\n[Total score]: \n{json.dumps(score_record, indent=4)}")

        save_preds_name = save_name + f"/{dataset_name}_predications.json"
        with open(f'{save_preds_name}', 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        save_score_name = save_name + f"/{dataset_name}_score.json"
        with open(save_score_name, 'w') as json_file:
            json.dump(score_record, json_file, indent=4)

        # print(f"\n\n[Total score]: \n{json.dumps(score_record, indent=4)}")

        print(f"\nCompleted ARC Subset {dataset_name}, pelase check {save_name}")
