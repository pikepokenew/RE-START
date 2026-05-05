import os
import glob
import json
import re

def merge_json_shards(file_pattern, output_path):
    """
    查找所有匹配 file_pattern 的 json 分片，
    按照 start_idx 排序，然后合并成一个文件保存到 output_path。
    """
    # 1. 查找所有匹配的文件
    shards = glob.glob(file_pattern)
    if not shards:
        print(f"Warning: No files found matching {file_pattern}")
        return

    print(f"Found {len(shards)} files. Parsing and sorting...")

    # 2. 从文件名中解析 start_idx 并排序
    # 文件名里带有类似 _start_0_end_625.json 的标识
    parsed_shards = []
    start_pattern = re.compile(r"_start_(\d+)_end_(-?\d+)\.json$")
    
    for shard in shards:
        match = start_pattern.search(shard)
        if match:
            start_idx = int(match.group(1))
            parsed_shards.append((start_idx, shard))
        else:
            print(f"Warning: Could not parse start_idx from {shard}. It will be appended at the end.")
            parsed_shards.append((float('inf'), shard))

    # 按 start_idx 排序，保证合起来后的数据顺序正确
    parsed_shards.sort(key=lambda x: x[0])

    # 3. 依次读取并合并
    merged_data = []
    for start_idx, shard_path in parsed_shards:
        print(f"Loading: {shard_path} (start_idx: {start_idx})")
        with open(shard_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: Data in {shard_path} is not a list. Appending as single item.")
                    merged_data.append(data)
            except json.JSONDecodeError:
                print(f"Error: {shard_path} is not a valid JSON file. Skipping.")

    # 4. 保存合并后的结果
    print(f"\nTotal items after merging: {len(merged_data)}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving merged data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
        
    print("Done!")


if __name__ == "__main__":
    # 你要匹配的文件特征 (支持 *)
    # PATTERN = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/harmful_questions/Qwen3-14B_sys_0_temp_0.6_n_1_None_prefix_0_start_*_end_*.json"
    # OUTPUT_FILE = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/harmful_questions/Qwen3-14B_sys_0_temp_0.6_n_1_None_prefix_0_merged.json"

    # PATTERN = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train/Qwen3-14B_sys_0_temp_0.6_n_1_RealSafe-R1_prefix_0_start_*_end_*.json"
    # OUTPUT_FILE = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train/Qwen3-14B_sys_0_temp_0.6_n_1_RealSafe-R1_prefix_0_merged.json"

    # PATTERN = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_RealSafe-R1_prefix_0_start_*_end_*.json"
    # OUTPUT_FILE = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_RealSafe-R1_prefix_0_merged.json"

    # PATTERN = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_start_*_end_*.json"
    # OUTPUT_FILE = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_DS_R1_14B/DeepSeek-R1-Distill-Qwen-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_merged.json"

    PATTERN = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_Qwen3_14B/Qwen3-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_start_*_end_*.json"
    OUTPUT_FILE = "/apdcephfs_jn3/share_535475/common/dellwu/RE-START/evaluate/results/wildjailbreak_train_Qwen3_14B/Qwen3-14B_sys_0_temp_0.6_n_1_self_align_v2_prefix_random_merged.json"
    
    # 合并后输出的文件名，这里把 start_xxx_end_xxx 这段去掉，作为最终完整版
    
    merge_json_shards(PATTERN, OUTPUT_FILE)
