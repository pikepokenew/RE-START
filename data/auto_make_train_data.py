#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# [AUTO] Training-data merger (CLI version of make_train_data.py)
#
# 本脚本是 data/make_train_data.py 的自动化副本（auto_ 前缀），过滤规则、messages
# 构建、think 字段拼接逻辑与原脚本完全一致，仅改为支持 CLI 参数：
#
#   python auto_make_train_data.py \
#       --file_list A.json B.json C.json \
#       --output_file /path/to/out.json
#
# 与原脚本的行为差异：
#   1. 强制要求 --file_list / --output_file 两个 CLI 参数（缺失直接报错退出）
#   2. 单文件缺失/为空：warning + 继续；所有文件都缺失：非零码退出
#   3. 结束时打印最终条目数与输出绝对路径，方便 workflow 日志追溯
# =============================================================================

import argparse
import json
import os
import sys
from typing import List, Dict, Any


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def filter_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter data.

    Drop:
      1. harmful request that is NOT refused (is_harmful_request=true, is_refusal_response=false)
      2. harmless request that IS refused  (is_harmful_request=false, is_refusal_response=true)
    """
    filtered_data = []

    for item in data:
        is_harmful = item.get('is_harmful_request', False)
        if type(is_harmful) == list:
            is_harmful = is_harmful[0]
        is_refusal = item.get('is_refusal_response', False)
        if type(is_refusal) == list:
            is_refusal = is_refusal[0]

        # Drop condition 1: harmful & not refused
        if is_harmful and not is_refusal:
            continue

        # Drop condition 2: harmless & refused
        if not is_harmful and is_refusal:
            continue

        filtered_data.append(item)

    return filtered_data


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build messages field:
    [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>...</think>\n\nresponse"}
    ]
    """
    messages = []

    question = item.get('question', '')
    if question:
        messages.append({
            "role": "user",
            "content": question
        })

    llm_response = item.get('llm_response', [])
    think = item.get('think', '')
    if isinstance(llm_response, list) and llm_response:
        assistant_content = "<think>\n{think}\n</think>\n\n{llm_response}".format(
            think=think[0], llm_response=llm_response[0]
        )
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
    elif isinstance(llm_response, str):
        assistant_content = "<think>\n{think}\n</think>\n\n{llm_response}".format(
            think=think, llm_response=llm_response
        )
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

    return messages


def process_data(file_list: List[str], output_file: str) -> int:
    """Main processing routine. Returns the number of output items."""
    all_filtered_data = []
    present_count = 0

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue
        if os.path.getsize(file_path) == 0:
            print(f"Warning: File is empty: {file_path}", file=sys.stderr)
            continue

        present_count += 1
        print(f"Processing {file_path}")
        data = load_json_data(file_path)
        filtered_data = filter_data(data)
        all_filtered_data.extend(filtered_data)
        print(f"  Original: {len(data)} items, Filtered: {len(filtered_data)} items")

    if present_count == 0:
        print("[FATAL][auto_make_train_data] all input files are missing or empty; nothing to merge.",
              file=sys.stderr)
        sys.exit(4)

    print(f"Total filtered data: {len(all_filtered_data)} items")

    # Output data format (identical to original make_train_data.py)
    output_data = []
    for item in all_filtered_data:
        messages = build_messages(item)
        output_item = {
            'messages': messages,
            'prefix': item.get('prefix', ''),
            # Keep raw fields for debugging
            'question': item.get('question', ''),
            'llm_response': item.get('llm_response', []),
            'is_harmful_request': item.get('is_harmful_request', False),
            'is_refusal_response': item.get('is_refusal_response', False),
        }
        output_data.append(output_item)

    # Ensure output dir exists
    output_abs = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    with open(output_abs, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Filtered data saved to: {output_abs}")
    print(f"[auto_make_train_data] OK: {len(output_data)} items -> {output_abs}")
    return len(output_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto version of make_train_data.py with strict CLI interface.",
    )
    parser.add_argument(
        "--file_list", nargs='+', required=True,
        help="List of input JSON files (space separated).",
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Absolute path to write the merged training JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_data(args.file_list, args.output_file)
