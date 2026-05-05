from datasets import load_dataset
import json

ds = load_dataset("openbmb/UltraFeedback")['train']
output_path = '/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/ultrafeedback_train_1k.json'

new_dataset = []
for item in ds:
    # import pdb; pdb.set_trace()
    new_dataset.append(item)

import random
# random.shuffle(new_dataset)
new_dataset = random.sample(new_dataset, 1000)
print("dataset size:{}".format(len(new_dataset)))
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)