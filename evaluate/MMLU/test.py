# from datasets import load_dataset
# import pandas as pd
# # ds = load_dataset("openai/MMMLU", "HI_IN")['test']
# # lanugage = "AR_XY" 
# # lanugage = "BN_BD" 
# lanugage = "JA_JP" 
# ds = load_dataset("openai/MMMLU", lanugage)['test']

# # new_data_list = []
# # for data in ds:


# total_dataset = {}
# for data in ds:
#     pass
#     subject = data['Subject']
#     Question = data['Question']
#     A     = data['A']
#     B     = data['B']
#     C     = data['C']
#     D     = data['D']
#     Answer  = data['Answer']

#     new_data = {
#         "Question": Question,
#         "A":    A,
#         "B":    B,
#         "C":    C,
#         "D":    D,
#         "Answer": Answer
#     }
#     if total_dataset.get(subject, None) == None:
#         total_dataset[subject] = []
#     total_dataset[subject].append(new_data)

# # for 

# for subject, dataset in total_dataset.items():
#     out_file = "data/MMMLU_{}/test/{}.csv".format(lanugage, subject)
#     df = pd.DataFrame(dataset)
#     df.to_csv(out_file)
# # import pdb; pdb.set_trace()




