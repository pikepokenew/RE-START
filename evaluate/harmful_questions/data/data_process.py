from datasets import load_dataset
import json

# ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", name = "default")['test']

# out_filename = "PKU-SafeRLHF_test.json"
# new_dataset = []
# for data in ds:
#     new_dataset.append(data)
#     pass

# with open(out_filename, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# ########################## PKU-Harmful_SFT_500_unseen ##########################
# train_file_path = "src/PKU-SafeRLHF_train.json"
# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# test_file_path = "src/PKU-SafeRLHF_test.json"
# with open(test_file_path, "r") as file:
#     test_dataset = json.load(file)

# train_data_dict = {}
# for data in train_dataset:
#     if train_data_dict.get(data['prompt'], None) == None:
#         train_data_dict[data['prompt']] = 1
    
#     train_data_dict[data['prompt']] = train_data_dict[data['prompt']] + 1

# out_file_path = "PKU-Harmful_SFT_500_unseen.json"
# unseen_test_data = []
# for data in test_dataset:
#     if train_data_dict.get(data['prompt'], None) == None and \
#         ( data['is_response_0_safe'] == False or data['is_response_1_safe'] == False ):
#         # unseen_test_data.append(data)
#         if data['is_response_0_safe'] == False:
#             output = data['response_0']
#         elif data['is_response_1_safe'] == False:
#             output = data['response_1']
#         new_data = {"instruction": data['prompt'], 
#                     "input": "", 
#                     "output": output,}
        
#         unseen_test_data.append(new_data)

# unseen_test_data = unseen_test_data[:500]

# print("unseen data size: {}".format(len(unseen_test_data)))

# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(unseen_test_data, f, ensure_ascii=False, indent=4)

# train_file_path = "src/PKU-SafeRLHF_train.json"
# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# test_file_path = "src/PKU-SafeRLHF_test.json"
# with open(test_file_path, "r") as file:
#     test_dataset = json.load(file)

# # num_samples = 500
# num_samples = 1500

# out_file_path = "PKU-Harmful_SFT_{}.json".format(num_samples)
# harmful_test_data = []
# for data in test_dataset:
#     if  ( data['is_response_0_safe'] == False or data['is_response_1_safe'] == False ):
#         # unseen_test_data.append(data)
#         if data['is_response_0_safe'] == False:
#             output = data['response_0']
#         elif data['is_response_1_safe'] == False:
#             output = data['response_1']
#         new_data = {"instruction": data['prompt'], 
#                     "input": "", 
#                     "output": output,}
        
#         harmful_test_data.append(new_data)

# harmful_test_data = harmful_test_data[:num_samples]

# print("unseen data size: {}".format(len(harmful_test_data)))

# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(harmful_test_data, f, ensure_ascii=False, indent=4)

# ##################### PKU-SafeRLHF v1 #########################
# train_file_path = "src/PKU-SafeRLHF_train.json"
# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# count = 0
# new_dataset = []
# for data in train_dataset:
#     unsafe_response_id = 1 - data['safer_response_id']
#     if data['is_response_{}_safe'.format(unsafe_response_id)] == False:
#         pass
#         new_dataset.append(data)
#         # count += 1

# out_file_path = "PKU-SafeRLHF_v1_train.json"
# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(new_dataset)))

# ##################### PKU-SafeRLHF v1 for kto #########################

# train_file_path = "src/PKU-SafeRLHF_train.json"
# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# count = 0
# new_dataset = []
# pos_count_max = 2500
# neg_count_max = 2500
# for data in train_dataset:
#     unsafe_response_id = 1 - data['safer_response_id']
#     if data['is_response_{}_safe'.format(unsafe_response_id)] == False:
        
#         messages = []
#         messages.append(
#             {
#                 "role": "user",
#                 "content": data['prompt']
#             }
#         )
#         if data['is_response_{}_safe'.format(data['safer_response_id'])] == True and pos_count_max > 0:
#             pass
#             messages.append(
#                 {
#                     "role": "assistant",
#                     "content": data['response_{}'.format(data['safer_response_id'])]
#                 }
#             )
#             new_data = {
#                 "messages": messages,
#                 "label": True,
#             }
#             pos_count_max -= 1
#             new_dataset.append(new_data)
#         elif neg_count_max > 0:
#             messages.append(
#                 {
#                     "role": "assistant",
#                     "content": data['response_{}'.format(unsafe_response_id)]
#                 }
#             )
#             new_data = {
#                 "messages": messages,
#                 "label": False,
#             }
#             neg_count_max -= 1
#             new_dataset.append(new_data)
#         # pass
        
        
#         # if pos_count + neg_count > 5000:
#         #     break
        
# # print("pos_count: {}".format(pos_count))
# # print("neg_count: {}".format(neg_count))
# # import random
# # random.shuffle(new_dataset)
# out_file_path = "PKU-SafeRLHF_v1_kto_train.json"
# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(new_dataset)))

# ##################### PKU-SafeRLHF v1 for Refusal #########################
# src_file_path = "/home/dwu/Immunization/evaluate/results/PKU-SafeRLHF_v1_train/Llama-2-7b-chat-hf_sys_1.json"
# with open(src_file_path, "r") as file:
#     src_dataset = json.load(file)

# kto_file_path = "/home/dwu/Immunization/data/PKU-SafeRLHF/PKU-SafeRLHF_v1_kto_train.json"
# with open(kto_file_path, "r") as file:
#     kto_dataset = json.load(file)

# new_dataset = []
# for data in kto_dataset:
#     pass
#     new_data = None
#     for src_data in src_dataset:
#         pass
#         if data['messages'][0]['content'] == src_data['prompt'] and (data['messages'][1]['content'] == src_data['response_0'] or data['messages'][1]['content'] == src_data['response_1']):
#             conversations = [
#                 {
#                     "from": "human",
#                     "value": src_data['prompt']
#                 }
#             ]
#             chosen = {
#                 "from": "gpt",
#                 "value": src_data['llm_response'],
#             }
#             rejected = {
#                 "from": "gpt",
#                 "value": src_data['response_{}'.format(1 - src_data['safer_response_id'])]
#             }

#             new_data = {
#                 "conversations": conversations,
#                 "chosen": chosen,
#                 "rejected": rejected,
#             }
#             new_dataset.append(new_data)
#             break

# out_file_path = "PKU-SafeRLHF_v1_refusal_train.json"
# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(new_dataset)))

# ##################### PKU-SafeRLHF v2 for kto #########################
# train_file_path = "raw_data/PKU-SafeRLHF_train.json"
# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# count = 0
# new_dataset = []
# # pos_count_max = 2500
# # neg_count_max = 2500
# for data in train_dataset:
#     unsafe_response_id = 1 - data['safer_response_id']
#     safer_response_id = data['safer_response_id']
#     # if data['is_response_{}_safe'.format(unsafe_response_id)] == False:
        
#     messages = []
#     messages.append(
#         {
#             "role": "user",
#             "content": data['prompt']
#         }
#     )
#     if data['is_response_{}_safe'.format(safer_response_id)] == True:
#         messages.append(
#             {
#                 "role": "assistant",
#                 "content": data['response_{}'.format(safer_response_id)]
#             }
#         )
#         new_data = {
#             "messages": messages,
#             "label": True,
#         }
#         new_dataset.append(new_data)
#     else:
#         messages.append(
#             {
#                 "role": "assistant",
#                 "content": data['response_{}'.format(unsafe_response_id)]
#             }
#         )
#         new_data = {
#             "messages": messages,
#             "label": False,
#         }

#         new_dataset.append(new_data)
# new_dataset = new_dataset[:5000]
# pos_count = 0
# neg_count = 0
# for data in new_dataset:
#     if data['label'] == True:
#         pos_count += 1
#     elif data['label'] == False:
#         neg_count += 1
# print("pos_count: {}".format(pos_count))
# print("neg_count: {}".format(neg_count))
# import random
# random.shuffle(new_dataset)
# out_file_path = "PKU-SafeRLHF_v2_kto_train.json"
# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(new_dataset)))
# ##################### PKU-SafeRLHF v2 for harmful fine-tuning #########################
# test_file_path = "raw_data/PKU-SafeRLHF_test.json"
# with open(test_file_path, "r") as file:
#     test_dataset = json.load(file)

# count = 0
# new_dataset = []
# # pos_count_max = 2500
# # neg_count_max = 2500
# for data in test_dataset:
#     unsafe_response_id = 1 - data['safer_response_id']
#     safer_response_id = data['safer_response_id']
#     # if data['is_response_{}_safe'.format(unsafe_response_id)] == False:
#     question = data['prompt']
#     output = data['response_{}'.format(unsafe_response_id)]    
#     new_data = {
#         "instruction": question,
#         "input": "",
#         "output": output,
#     }
#     new_dataset.append(new_data)
# import random
# random.shuffle(new_dataset)

# for max_samples in [500, 1000, 1500, -1]:
#     save_dataset = new_dataset[:max_samples]
#     out_file_path = "PKU-SafeRLHF_v2_Harmful_SFT_{}.json".format(max_samples)
#     with open(out_file_path, 'w', encoding='utf-8') as f:
#         json.dump(save_dataset, f, ensure_ascii=False, indent=4)

#     print("dataset size: {}".format(len(save_dataset)))
# ##################### PKU-SafeRLHF v2 for kto tulu2/llama2-self-distill #########################
# train_file_path = "rewrite/tulu-2-7b_prefix_256_max_rnd_10_sys_0_wildguard_chat_template_1_as_judge.json"
# out_file_path = "PKU-SafeRLHF_v2_kto_tulu2_distill_train.json"

# train_file_path = "rewrite/tulu-2-7b_full_harmful_sft_lr_2e-5_cosine_epoch_1_prefix_0_max_rnd_10_sys_0_wildguard_chat_template_1_as_judge.json"
# out_file_path = "PKU-SafeRLHF_v2_kto_tulu2_distill_v3_train.json"

# with open(train_file_path, "r") as file:
#     train_dataset = json.load(file)

# count = 0
# new_dataset = []
# # pos_count_max = 2500
# # neg_count_max = 2500
# for idx, data in enumerate(train_dataset):
#     # unsafe_response_id = 1 - data['safer_response_id']
#     # safer_response_id = data['safer_response_id']
#     # if data['is_response_{}_safe'.format(unsafe_response_id)] == False:
        
#     messages = []
#     messages.append(
#         {
#             "role": "user",
#             "content": data['question']
#         }
#     )
#     if idx % 2 == 0:
#         messages.append(
#             {
#                 "role": "assistant",
#                 "content": data['generated_response'].strip(" \n")
#             }
#         )
#         new_data = {
#             "messages": messages,
#             "label": True,
#         }
#         new_dataset.append(new_data)
#     else:
#         messages.append(
#             {
#                 "role": "assistant",
#                 "content": data['harmful_response']
#             }
#         )
#         new_data = {
#             "messages": messages,
#             "label": False,
#         }

#         new_dataset.append(new_data)


# with open(out_file_path, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(new_dataset)))


# ######################## HH-RLHF #########################

# from datasets import load_dataset

# ds = load_dataset("Anthropic/hh-rlhf")['train']
# drop_count = 0
# new_dataset = []
# for data in ds:
#     new_conversations = {}
#     for response_type, message in data.items():
#         new_conversations[response_type] = []
#         # print("#" * 50)
#         # print(message)
#         conversations = message.split("\n\nHuman:")
#         for round in conversations:

#             if round == "":
#                 continue
#             sentences = round.split("\n\nAssistant:")
#             # if len()
#             # print(sentences)
#             if len(sentences) != 2:
#                 drop_count += 1
#             if len(sentences) < 2:
#                 continue
#             # import pdb; pdb.set_trace()
#             sentences = sentences[:2]
#             # assert len(sentences) == 2
#             new_conversations[response_type].append(
#                 {
#                     "from": "human",
#                     "value": sentences[0].strip(" \n")
#                 }
#             )
#             new_conversations[response_type].append(
#                 {
#                     "from": "gpt",
#                     "value": sentences[1].strip(" \n")
#                 }
#             )
#             # for sent in sentences:
#     new_dataset.append(new_conversations)
#     # import pdb; pdb.set_trace()


# hh_rlhf_dataset = []

# for data in new_dataset:
#     conversations = data['chosen'][:-1]
#     chosen = data['chosen'][-1]
#     rejected = data['rejected'][-1]
#     new_data = {
#         "conversations": conversations,
#         "chosen": chosen,
#         "rejected": rejected,
#     }
#     hh_rlhf_dataset.append(new_data)

# print("drop_count: {}".format(drop_count))
# save_name = "hh-rlhf.json"

# with open(save_name, 'w', encoding='utf-8') as f:
#     json.dump(hh_rlhf_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(hh_rlhf_dataset)))


# import json
# src_file_path = "/home/dwu/Immunization/evaluate/results/data/hh-rlhf_Meta-Llama-Guard-2-8B_as_judge_labelled.json"
# with open(src_file_path, "r") as file:
#     train_dataset = json.load(file)

# hh_rlhf_dataset = []
# for data in train_dataset:
#     # data['chosen']
#     if data['chosen']['label'] == "Unharmful" and data['rejected']['label'] == "Harmful":
#         new_data = data
#         del new_data['rejected']['label']
#         del new_data['chosen']['label']
#         hh_rlhf_dataset.append(new_data)
# save_name = "hh-rlhf_v2.json"

# with open(save_name, 'w', encoding='utf-8') as f:
#     json.dump(hh_rlhf_dataset, f, ensure_ascii=False, indent=4)

# print("dataset size: {}".format(len(hh_rlhf_dataset)))

# ############################ OASST2 ##############################
# parsing OA data files with oasst_data helpers
# from oasst_data import read_message_trees, visit_messages_depth_first, ExportMessageNode

# messages: list[ExportMessageNode] = []

# input_file_path = "2023-11-05_oasst2_all.messages.jsonl.gz"
# for tree in read_message_trees(input_file_path):
#     if tree.prompt.lang not in ["en","es"]:     # filtering by language tag (optional)
#         continue

#     # example use of depth first tree visitor help function
#     visit_messages_depth_first(tree.prompt, visitor=messages.append, predicate=None)

# ############################### MaliciousInstruct ###############################
# import json
# src_file_path = "MaliciousInstruct.txt"
# new_dataset = []
# with open(src_file_path, "r") as file:
#     for line in file:  # 逐行读取文件内容
#         # print(line.strip())  # 使用strip()去除每行末尾的换行符
#         new_dataset.append(line.strip())
#     # train_dataset = json.load(file)
# save_name = "MaliciousInstruct.json"
# with open(save_name, 'w', encoding='utf-8') as f:
#     json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# ######################### StrongREJECT #########################
import json
import pandas as pd
src_file_path = "strongreject_dataset.csv"
new_dataset = []
df = pd.read_csv(src_file_path)

for idx, row in df.iterrows():
    new_data = {
        "prompt": row['forbidden_prompt'],
        "category": row['category'],
        "source": row['source'],
    }
    new_dataset.append(new_data)
# with open(src_file_path, "r") as file:
#     for line in file:  # 逐行读取文件内容
#         # print(line.strip())  # 使用strip()去除每行末尾的换行符
#         new_dataset.append(line.strip())
    # train_dataset = json.load(file)

save_name = "StrongREJECT.json"
with open(save_name, 'w', encoding='utf-8') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)