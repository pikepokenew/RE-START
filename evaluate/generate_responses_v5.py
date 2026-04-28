import os
import json
import argparse
import sys
import threading
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取当前路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append("/home/dwu/Immunization")

# =========================================================================
# 用户配置区域
# =========================================================================
INJECTION_MAPPING = {
    "step 1": " [System Check: Verified]\n",
    "Analysis:": "\n[Guideline: Be extremely critical in the following analysis]\n",
    "Conclusion:": "\n[Safety Constraint: Ensure the answer is harmless]\n",
}

MAX_INJECTION_LOOPS = 10 
# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model path or name', type=str, required=True)
parser.add_argument('--api_base', help='vllm api base url', type=str, default="http://localhost:8000/v1")
parser.add_argument('--api_key', help='api key', type=str, default="EMPTY")
parser.add_argument('--save_path', help='save directory', type=str, default='evaluate/results')
parser.add_argument('--save_name', help='custom save filename', type=str, default=None)
parser.add_argument('--dataset', help='path to dataset json', required=True, type=str)
parser.add_argument('--num_samples', help='number of samples', type=int, default=-1)
parser.add_argument('--start_idx', help='start index', type=int, default=0)
parser.add_argument('--n_generation', help='generations per prompt', type=int, default=1)
parser.add_argument('--max_new_tokens', help='max new tokens', type=int, default=512)
parser.add_argument('--temperature', help='temperature', type=float, default=0.0)
parser.add_argument('--top_p', help='top_p', type=float, default=1.00)
parser.add_argument('--seed', help='seed', type=int, default=42)
# [新增参数] 线程数，建议设置为 10-50，取决于你的 vLLM 显存吞吐量
parser.add_argument('--num_threads', help='number of concurrent threads', type=int, default=16) 
args = parser.parse_args()


def process_data(dataset_name, nsamples, start_idx=0):
    """
    数据读取与预处理函数
    """
    with open(dataset_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    new_dataset = []
    
    # 适配不同格式的 JSON 数据集
    if "catqa" in dataset_name and isinstance(dataset, dict):
        for topic, data_list in dataset.items():
            for subtopic, sub_data_list in data_list.items():
                for data in sub_data_list:
                    new_dataset.append({'question': data, 'topic': topic, 'subtopic': subtopic})
    elif isinstance(dataset, list):
        for data in dataset:
            if isinstance(data, str):
                new_dataset.append({'question': data})
            elif isinstance(data, dict):
                question = None
                if data.get("question"):
                    question = data["question"]
                elif data.get("prompt"):
                    question = data["prompt"]
                elif data.get("instruction"):
                    question = data["instruction"]
                    if data.get("input"):
                        question += "\n" + data["input"]
                elif data.get("messages"):
                    question = data["messages"][0]["content"]
                elif data.get("adversarial") is not None:
                    question = data["adversarial"] if data["adversarial"] else data.get("vanilla")
                elif data.get("qA_pairs", None) != None:
                    question = data['qA_pairs'][0]['Q']
                if question:
                    item = data.copy()
                    item['question'] = question
                    new_dataset.append(item)
    
    if nsamples == -1:
        nsamples = len(new_dataset)
    return new_dataset[start_idx: start_idx + nsamples]


def generate_via_raw_completion(client, model_name, raw_prompt, injection_map, max_tokens, temperature, top_p, seed):
    """
    核心函数：流式生成 + 注入逻辑
    """
    current_prompt = raw_prompt
    full_generated_content = ""
    injection_map = {
        "policy": " (1. Comply with applicable laws; 2. Don't provide information hazards; 3. Respect creators and their rights; 4. Protect people's privacy; 5. Don't respond with NSFW content)",
        "policies": " (1. Comply with applicable laws; 2. Don't provide information hazards; 3. Respect creators and their rights; 4. Protect people's privacy; 5. Don't respond with NSFW content)",
    }
    triggers = list(injection_map.keys())
    has_injected = False 

    for loop_i in range(MAX_INJECTION_LOOPS + 1):
        # 简单处理：每次都给最大 token，防止截断。
        # 如果需要严格控制总长度，这里可以用 max_tokens - len(generated) 计算
        current_max = max_tokens 

        try:
            stream = client.completions.create(
                model=model_name,
                prompt=current_prompt,
                max_tokens=current_max,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                stream=True
            )
            
            buffer_text = ""
            matched_trigger = None
            
            for chunk in stream:
                delta = chunk.choices[0].text
                if delta:
                    buffer_text += delta
                    
                    if not has_injected:
                        for trigger in triggers:
                            if buffer_text.endswith(trigger):
                                matched_trigger = trigger
                                break
                        if matched_trigger:
                            stream.close()
                            break
            
            full_generated_content += buffer_text
            current_prompt += buffer_text

            if matched_trigger and loop_i < MAX_INJECTION_LOOPS:
                injection_text = injection_map[matched_trigger]
                
                full_generated_content += injection_text
                current_prompt += injection_text
                has_injected = True
                continue 
            else:
                break

        except Exception as e:
            print(f"Generation Error at loop {loop_i}: {e}")
            break
            
    return full_generated_content


def process_single_sample(idx, data, tokenizer, client, args):
    """
    [新增] 单个样本的处理逻辑，用于多线程调用
    """
    try:
        question = data['question']
        # if data.get("question", None) != None:
        #     question = data['question']
        # elif data.get("qA_pairs", None) != None:
        #     question = data['qA_pairs'][0]['Q']
        messages = [{"role": "user", "content": question}]
        
        # 1. 构造 Prompt
        raw_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. 生成 (支持 n_generation)
        generated_responses = []
        for _ in range(args.n_generation):
            resp = generate_via_raw_completion(
                client=client,
                model_name=args.model,
                raw_prompt=raw_prompt, 
                injection_map=INJECTION_MAPPING,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed
            )
            generated_responses.append(resp)

        # 3. 结果解析
        new_item = data.copy()
        new_item["llm_response"] = []
        new_item['think'] = []
        
        for r in generated_responses:
            if "</think>" in r:
                parts = r.split("</think>")
                think_content = parts[0].replace("<think>", "").strip()
                content = "".join(parts[1:]).strip()
                new_item['think'].append(think_content)
                new_item["llm_response"].append(content)
            else:
                new_item["llm_response"].append(r.strip())
        
        # 返回索引和结果，以便后续排序
        return idx, new_item

    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return idx, None


if __name__ == '__main__':
    print(f"\nConfiguration:")
    print(f"*{'-'*10}*")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"*{'-'*10}*\n")

    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    try:
        tokenizer_path = "/home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except:
             print("Fatal Error: Raw completion mode requires a local tokenizer.")
             sys.exit(1)

    # 2. 初始化 OpenAI 客户端
    # OpenAI Client 是线程安全的
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=600.0, # 增加超时时间，防止并发高时排队超时
        max_retries=3
    )

    # 3. 数据准备
    dataset = process_data(args.dataset, args.num_samples, args.start_idx)
    print(f"Loaded {len(dataset)} samples.")
    
    # 4. 路径准备
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.save_name:
        save_name = f"{args.save_path}/{args.save_name}"
    else:
        model_short = args.model.split("/")[-1]
        data_short = args.dataset.split("/")[-1].replace(".json","")
        save_name = f'{args.save_path}/{data_short}/{model_short}_{args.start_idx}-{args.start_idx + len(dataset)}_temp_{args.temperature}.json'
    
    if os.path.dirname(save_name) and not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))

    print(f"Generating responses with {args.num_threads} threads... Saving to: {save_name}\n")
    
    # 结果容器
    final_results = [None] * len(dataset) # 预分配空间，用于按索引位置填入
    file_lock = threading.Lock() # 用于文件写入锁
    
    # 使用 ThreadPoolExecutor 并发请求
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # 提交任务：传入 (index, data, tokenizer, client, args)
        future_to_idx = {
            executor.submit(process_single_sample, i, dataset[i], tokenizer, client, args): i 
            for i in range(len(dataset))
        }

        # 使用 as_completed 实时获取结果
        for future in tqdm(as_completed(future_to_idx), total=len(dataset)):
            idx, result = future.result()
            
            if result is not None:
                final_results[idx] = result # 按原始索引存放
            else:
                # 如果失败，保留原始数据作为占位，避免列表错位
                final_results[idx] = dataset[idx]
                final_results[idx]["error"] = "Processing failed"

            # [可选] 实时保存逻辑
            # 为了性能，建议每完成 N 个或者是最后统一保存。
            # 这里为了安全起见，每次有结果都存一次，但在高并发下可能会有点拖慢 IO。
            # 改进：我们只在最后保存，或者每完成 10 个保存一次。
            
            # 这里我保留你原来的实时保存逻辑，加上了锁
            with file_lock:
                 # 注意：实时 dump 整个列表比较慢，但为了数据安全...
                 # 为了避免写 None 到文件里，我们过滤掉还没完成的
                 current_valid_outputs = [res for res in final_results if res is not None]
                 with open(save_name, 'w', encoding='utf-8') as f:
                     json.dump(current_valid_outputs, f, ensure_ascii=False, indent=4)

    # 最终完整保存 (确保排序正确)
    print("\nFinalizing save...")
    valid_results = [res for res in final_results if res is not None]
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=4)

    print(f"\nCompleted. Please check {save_name}")