"""合并 run_hint_pipeline.sh 产生的分片文件。

用法示例：
  python merge_shards.py \
      --shard_dir evaluate/results/wildjailbreak_train_Qwen3-14B/shards \
      --pattern 'step2_shard*_prefix_random_wildguard.json' \
      --output  evaluate/results/wildjailbreak_train_Qwen3-14B/step2_merged.json

  python merge_shards.py \
      --shard_dir evaluate/results/wildjailbreak_train_Qwen3-14B/shards \
      --pattern 'step4_shard*_with_hint_16_wildguard.json' \
      --output  evaluate/results/wildjailbreak_train_Qwen3-14B/step4_merged.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import List


def _sort_key(path: str) -> int:
    """按 shardN 中的 N 做自然排序，保证合并顺序与原 global range 对齐。"""
    m = re.search(r"shard(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def merge(shard_dir: str, pattern: str, output: str, allow_empty: bool = False) -> None:
    files = sorted(glob.glob(os.path.join(shard_dir, pattern)), key=_sort_key)
    if not files:
        msg = f"No shard files matched: {shard_dir}/{pattern}"
        if allow_empty:
            print(f"[merge] {msg} (allow_empty, writing empty list)")
            os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
            if output.endswith(".jsonl"):
                open(output, "w", encoding="utf-8").close()
            else:
                with open(output, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)
            return
        raise FileNotFoundError(msg)

    print(f"[merge] matched {len(files)} files:")
    for p in files:
        print(f"  - {p}")

    merged: List[dict] = []
    for p in files:
        if os.path.getsize(p) == 0:
            print(f"[warn] {p} is empty, skipped.")
            continue
        with open(p, "r", encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
            else:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError(f"{p} does not contain a JSON list.")
                merged.extend(data)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    if output.endswith(".jsonl"):
        with open(output, "w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"[merge] total {len(merged)} items -> {output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--pattern", required=True,
                   help="glob pattern, e.g. 'step2_shard*_wildguard.json'")
    p.add_argument("--output", required=True)
    p.add_argument("--allow_empty", action="store_true",
                   help="若没有匹配到任何分片文件，则写出空列表而不是报错。")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.shard_dir, args.pattern, args.output, allow_empty=args.allow_empty)
