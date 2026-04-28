import json
import time
from openai import OpenAI
import argparse
import pandas as pd
from tqdm import tqdm
import os
import argparse
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from vllm import LLM, SamplingParams
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add

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
xfinder_model_name = "xFinder-qwen1505"
def get_prompts(dataset):
    
    prompt_list = []
    prompts_map_indices = []
    for idx, data in enumerate(dataset):
        if "None" in data['correct_answers']:
            continue
        for j in range(len(data['llm_response'])):
            # import pdb; pdb.set_trace()
            question = data['instruction']
            llm_output = data['llm_response'][j]
            standard_answer_range = data['correct_answers']

            extract_formatted_query = xfinder_query_template.format(question = question, llm_output = llm_output, standard_answer_range=str(standard_answer_range))
            xfinder_prompt = XFINDER_PROMPT_TEMPLATE[xfinder_model_name].format(system=XFINDER_SYSTEM_PROMPT, input=extract_formatted_query)
            
            # import pdb; pdb.set_trace()
            prompt_list.append(xfinder_prompt)
            prompts_map_indices.append(idx)
    return prompt_list, prompts_map_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", help='json path where responses are saved', type=str)
    parser.add_argument("--num_samples", help='json path where responses are saved', type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", help='tensor_parallel_size', type=int, default=1)
    parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
    parser.add_argument('--key', help='key', type=str, required=False, default='llm_response')
    parser.add_argument('--over_safety', help='over_safety', type=int, required=False, default=0)

    args = parser.parse_args()

    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")

    dataset = json.load(open(args.response_file, 'r'))
    if args.num_samples != -1:
        dataset = dataset[:args.num_samples]

    prompts, prompts_map_indices = get_prompts(dataset=dataset)
    xfinder_model_name = "xFinder-qwen1505"
    key_answer_type = "short text"

    model_name_or_path = "/home/dwu/local_models/{}".format(xfinder_model_name)
    xfinder = LLM(model_name_or_path, tensor_parallel_size = args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/dwu/local_models/{}".format(xfinder_model_name), trust_remote_code=True
    )
    stop_words = []
    stop_words.append(tokenizer.eos_token)
    stop_words.append("<|eot_id|>")
    outputs = xfinder.generate(prompts, SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=32,
                n=1,
                stop=stop_words,
    ))
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs_list = [output.outputs[0].text for output in outputs]

    for idx in range(len(outputs_list)):
        pred = outputs_list[idx]
        data_idx = prompts_map_indices[idx]
        answers = dataset[data_idx]['correct_answers']
        answers = [ans.strip("\n") for ans in answers]
        # import pdb; pdb.set_trace()
        if pred in answers:
            is_correct = True
        else:
            is_correct = False
        if dataset[data_idx].get("is_correct", None) == None:
            dataset[data_idx]['is_correct'] = []
        dataset[data_idx]['is_correct'].append(is_correct)
    
    save_name = args.response_file.replace(".json", "_check_ans.json")
    with open(f'{save_name}', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print("save result file: {}".format(save_name))