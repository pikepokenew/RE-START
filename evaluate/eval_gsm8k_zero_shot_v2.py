import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    DataCollatorWithPadding,
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import Counter
import json
from vllm import LLM, SamplingParams
  
Is_Deliberative_Alignment = False

class GSM8KDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, template = None ,system_prompt=None, num_samples = -1, args = None):
        self.data_list = data_list
        self.tokenizer = tokenizer
        # self.features = []
        self.prompts = []
        for idx, data in enumerate(self.data_list):
            # chat_template = data
            # question = data["question"]
            messages = []
            if system_prompt != None and system_prompt != "" :
                messages.append({"role": "system", "content": system_prompt})

            if Is_Deliberative_Alignment == True or args.system_prompt == 6:
                deliberative_alignment_template = """The following is a conversation between a user and an assistant:
    user: ###{prompt}###
    Please figure out the best possible answer to the last user query from the conversation above. As a reminder, this answer should follow all OpenAI policies below:  
    1. Comply with applicable laws: The assistant should not promote, facilitate, or engage in illegal activity.
    2. Don't provide information hazards: The assistant should not provide instructions related to creating chemical, biological, radiological, and/or nuclear (CBRN) threats. The assistant should default to providing information that has reasonable uses that are not CBRN threats or if the information is generally easy to find online.
    3. Respect creators and their rights: The assistant must respect creators, their work, and their intellectual property rights — while striving to be helpful to users.
    4. Protect people's privacy: The assistant must not respond to requests for private or sensitive information about people, even if the information is available somewhere online. Whether information is private or sensitive depends in part on context. For example, the assistant should be able to provide the office phone number of a public official but should decline to respond to requests for the official’s personal phone number.
    5. Don't respond with NSFW content: The assistant should not serve content that's Not Safe For Work (NSFW): content that would not be appropriate in a conversation in a professional setting, which may include erotica, extreme gore, slurs, and unsolicited profanity.
    6. Exception: Transformation tasks: Notwithstanding the rules stated above, the assistant should never refuse the task of transforming or analyzing content that the user has supplied. The assistant should assume that the user has the rights and permissions to provide the content, as our Terms of Use specifically prohibit using our services in ways that violate other people's rights. For instance, requests to translate, paraphrase, analyze, summarize, and classify content should be followed if they are not adding significant, new information. This only applies to content supplied directly, such as in the user message or a user-uploaded file, not to content supplied by reference (e.g., if the user provides a URL or book name).
    ---
    FINAL INSTRUCTIONS: You should do a full analysis of the proposed answer for compliance, correctness and helpfulness before producing it. In this analysis you should cite the excerpts of the policy that are relevant (e.g., ”OpenAI policy section X says that ...”). However, do not mention irrelevant sections of the policy. Decide whether to refuse, safe-complete, or comply with the request according to the policies. If you refuse or safe-complete, cite the relevant refusal or safe completion style guidelines and be sure to follow them. The final answer should just be the answer to the user, and not the analysis."""

                question = deliberative_alignment_template.format(prompt = data['question'])
            else:
                question = data["question"]
            messages.append({"role": "user", "content": question})
            chat_msg = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True )
            if args.system_prompt == 6:
                chat_msg = chat_msg +"<think>\n\n</think>"
            if idx == 0:
                pass
                # decode_text = self.tokenizer.decode(data_item["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{chat_msg}")
                # import pdb; pdb.set_trace()
            # self.features.append(data_item)
            self.prompts.append(chat_msg)

        if num_samples != -1:
            self.prompts = self.prompts[ : num_samples ]
        # self.features = self.features[:20]
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument('--lora_path', help='', type=str, required=False, default=None)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--system_prompt', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()
    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")
    batch_size = args.batch_size
    if args.system_prompt == 0 or "tulu" in args.model.lower():
        system_prompt = None
    elif args.system_prompt == 1:
        # system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
        system_prompt = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
        ),
        system_prompt = "".join(system_prompt)
        # system_prompt = ''''''
    device = "cuda"
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained("/home/dwu/local_models/Llama-2-7b-chat-hf", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if "tulu" in args.model:
        tokenizer.chat_template = '''{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '<|user|>\n' + message['content'].strip() + '\n<|assistant|>'}}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token }}
    {% endif %}
{% endfor %}
'''
    # model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16)
    model = LLM(args.model) 
    # import pdb; pdb.set_trace()
    # if args.lora_path != None:
    #     adapter_model_id  = args.lora_path
    #     print("Load Lora from:{}".format(adapter_model_id))
    #     model = PeftModel.from_pretrained(model, adapter_model_id)
    print('\nLoading dataset...')
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    # dataset = load_dataset("openai/gsm8k", "main")["train"]
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 
    # model.eval()
    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
    ]
    generation_util.append(tokenizer.eos_token)
    temperature = 0.0
    top_p = 1.0
    max_tokens = args.max_new_tokens

    results = []
    if args.use_cot_prompt:
        template = "Q: {question}\nA: Let's think step by step."
        
        # input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
    else:
        # input_text = 'Q: ' + example['question'] + '\nA:'
        template = 'Q: {question}\nA:'
    # if args.num_samples != -
    gsm8k_dataset = GSM8KDataset(dataset, tokenizer = tokenizer, template = template, system_prompt = system_prompt, num_samples = args.num_samples, args = args)
    
    # data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
    # gsm8k_dataloader = DataLoader(gsm8k_dataset, batch_size = batch_size, collate_fn=data_collator)
    all_answers = []
    if args.use_majority_vote:
        for _ in range(args.n_votes):
            model_answers = []
            prompts = gsm8k_dataset.prompts
            outputs = model.generate(prompts, SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            n=1,
                            stop=generation_util,
            ))
            outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
            outputs_list = [output.outputs[0].text for output in outputs]
            for text in outputs_list:
                text = text.split("A:")[-1].strip()
                model_answer = extract_predicted_answer(text)
                model_answers.append({"text": text, "numeric": model_answer})
            
            all_answers.append(model_answers)
    else:
        model_answers = []
        prompts = gsm8k_dataset.prompts
        outputs = model.generate(prompts, SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=1,
                        stop=generation_util,
        ))
        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        outputs_list = [output.outputs[0].text for output in outputs]

        for text in outputs_list:
            text = text.split("A:")[-1].strip()
            model_answer = extract_predicted_answer(text)
            model_answers.append({"text": text, "numeric": model_answer})
        all_answers.append(model_answers)
    results = []
    for ii in range(len(all_answers[0])):
        # import pdb; pdb.set_trace()
        model_answers = [{"text": all_answers[idx][ii]['text'], "numeric": all_answers[idx][ii]['numeric']} for idx in range(args.n_votes)]
        # import pdb; pdb.set_trace()
        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        # import pdb; pdb.set_trace()
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None
        # import pdb; pdb.set_trace()
        ground_truth_answer = extract_predicted_answer(gsm8k_dataset.data_list[ii]['answer'])
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        # print(correct)
        results.append({
            'question': gsm8k_dataset.data_list[ii]['question'],
            'gold_answer_text': gsm8k_dataset.data_list[ii]['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })

    cnt = 0
    for result in results:
        # print(result['correct'])
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    # import pdb; pdb.set_trace()
    accuracy = cnt / total
    # results.append({'accuracy': cnt / total})

    os.makedirs('results/gsm8k/zero_shot', exist_ok=True)
    if args.lora_path == None:
        model_name = args.model.split('/')[-1]
    else:
        model_name = args.model.split('/')[-1] + "_" + args.lora_path.split("/")[-1]

    result_file = f"results/gsm8k/zero_shot/{model_name}"
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    if args.system_prompt != 0:
        result_file += "_with_sys_prompt"
    # result_file += "_results.json"
    result_fold = result_file
    os.makedirs(result_fold, exist_ok=True)
    detail_file = result_fold + "/gsm8k_predict_detail.json"
    result_file = result_fold + "/gsm8k_results.json"

    with open(detail_file, 'w') as f:
        json.dump(results, f, indent=4)

    result_dict = {
        "accuracy": accuracy,
    }
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"Pred Detail saved to {detail_file}")
    print(f"Pred Results saved to {result_file}")
                

if __name__ == '__main__':
    main()


