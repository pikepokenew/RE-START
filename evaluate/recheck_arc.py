
from vllm import LLM, SamplingParams
import argparse
import json
import os
from typing import List, Dict, Any
import re
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

def parse_args():
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 包含所有解析后的参数的对象。
    """
    parser = argparse.ArgumentParser(
        description="读取指定的 JSON 文件，并将其内容打印到控制台。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 必需的位置参数：JSON 文件路径
    parser.add_argument(
        '--file_path', 
        type=str, 
        help="要读取的 JSON 文件路径。"
    )
    
    # 可选的布尔参数：是否打印额外的调试信息
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help="启用详细模式，打印额外的处理信息和异常追踪。"
    )
    parser.add_argument('--tensor_parallel_size', help='tensor_parallel_size', type=int, required=False, default=1)
    # 可选参数：可以根据需要添加更多参数，例如 encoding 或 default_value
    # parser.add_argument(
    #     '-e', '--encoding',
    #     type=str,
    #     default='utf-8',
    #     help="指定文件的编码格式 (默认: utf-8)。"
    # )
    
    return parser.parse_args()

def read_json_file(file_path: str, verbose: bool = False):
    """
    读取并解析指定的 JSON 文件。

    Args:
        file_path (str): JSON 文件的完整路径。
        verbose (bool): 是否打印详细信息。
        
    Returns:
        dict | list | None: 成功解析的 JSON 数据，如果发生错误则返回 None。
    """
    
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件未找到 - '{file_path}'")
        return None
        
    if verbose:
        print(f"ℹ️ 正在尝试读取文件: {file_path}")

    try:
        # 使用 'with' 语句确保文件在使用后被正确关闭
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用 json.load() 从文件中直接读取和解析 JSON 数据
            data = json.load(f)
            
        if verbose:
            print(f"✅ 文件读取成功。数据类型: {type(data).__name__}")
        
        return data

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: 文件 '{file_path}' 的内容不是有效的 JSON 格式。")
        if verbose:
            print(f"   详细信息: {e}")
        return None
        
    except Exception as e:
        print(f"❌ 发生未知错误: 在读取或处理文件 '{file_path}' 时出现问题。")
        if verbose:
            # 打印更详细的异常信息，以便调试
            import traceback
            traceback.print_exc()
        return None

def main():
    """
    程序的入口点。解析参数并调用主函数。
    """
    args = parse_args()
    
    # 调用读取函数
    json_data = read_json_file(args.file_path, args.verbose)
    xfinder_model_name = "xFinder-qwen1505"
    key_answer_type = "alphabet option"

    xfinder_inputs = []
    for idx in  range(len(json_data)):
        # import pdb; pdb.set_trace()
        question = json_data[idx]['question']
        input_prompt = question
        standard_answer_range = []
        for option, text in zip(json_data[idx]['choices']['label'], json_data[idx]['choices']['text']):
            input_prompt = input_prompt + f"\n({option}):{text}"
            standard_answer_range.append([option, text])
        # import pdb; pdb.set_trace()
        llm_output = json_data[idx]['response'].split("</think>")[-1].strip("\n ")


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

    cumulative_score = 0
    # import pdb; pdb.set_trace()
    max_score = len(json_data)
    for idx, responses in enumerate(xfinder_outputs_list):
        # import pdb; pdb.set_trace()

        answer = xfinder_outputs_list[idx]
        preds = extract_answer(answer, json_data[idx]['choices']['label'])

        correct_answer = json_data[idx]['answerKey']
        # import pdb; pdb.set_trace()
        # question = orig_que[idx]
        question = json_data[idx]['question']

        num_answers = len(preds)
        if correct_answer in preds:
            # 如果正确答案在预测中，得分是 1/N
            score = 1.0 / num_answers
        else:
            # 如果正确答案不在预测中，得分为 0
            score = 0.0
        cumulative_score += score
        
        json_data[idx]['score'] = score
        # response = [{'prompt':question, 'response':response, ""}]

        # result_record.append(data_result)

    score_record = {"max_score": max_score, "cumulative_score": cumulative_score, "acc": cumulative_score / max_score * 100.0}

    print(f"\n\n[Total score]: \n{json.dumps(score_record, indent=4)}")

    # save_preds_name = save_name + f"/{dataset_name}_predications.json"
    # with open(f'{save_preds_name}', 'w', encoding='utf-8') as f:
    #     json.dump(result_record, f, ensure_ascii=False, indent=4)

    # save_score_name = save_name + f"/{dataset_name}_score.json"
    # with open(save_score_name, 'w') as json_file:
    #     json.dump(score_record, json_file, indent=4)
    # print(f"\nCompleted ARC Subset {dataset_name}, pelase check {save_name}")


if __name__ == '__main__':
    result = main()
