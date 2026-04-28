from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.ReNeLLM_ding_2023 import ReNeLLM
from easyjailbreak.attacker.CodeChameleon_2024 import CodeChameleon
from easyjailbreak.attacker.DeepInception_Li_2023 import DeepInception
from easyjailbreak.attacker.Cipher_Yuan_2023 import Cipher
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
# target_model = from_pretrained(model_name_or_path='/home/dwu/local_models/Llama-2-7b-chat-hf',
#                                model_name='llama-2')
# vllm serve /home/share/models/Llama-2-7b-chat-hf --port 8000 --api-key token-abc123
attack_method = "Chipher"


stop_words = ["<|eot_id|>"]
target_model = OpenaiModel(model_name='/home/share/models/Llama-2-7b-chat-hf',
                           base_url = "http://localhost:8000/v1",
                         api_keys='token-abc123',
                         chat_template="llama-2",
                         generation_config = {"stop": stop_words})

attack_model = OpenaiModel(model_name='gpt-4o',
                           base_url = "https://api.chatanywhere.tech/v1",
                         api_keys='sk-p8oxw3nIBVTsNpLUfMvxiNh0es5cK0PsMKKgXs4AmPeqU9HI')
eval_model = OpenaiModel(model_name='gpt-4o',
                            base_url = "https://api.chatanywhere.tech/v1",
                         api_keys='sk-p8oxw3nIBVTsNpLUfMvxiNh0es5cK0PsMKKgXs4AmPeqU9HI')

num_samples = 520
max_workers = 520

dataset = JailbreakDataset('AdvBench', num_samples = num_samples)
# dataset = JailbreakDataset('/home/dwu/Immunization/evaluate/results/alpaca_data_en_52k/alpaca_data_en.json', num_samples = num_samples)

save_path = "AdvBench_{}_attack_result2.jsonl".format(attack_method)
if attack_method == "ReNeLLM":
  attacker = ReNeLLM(attack_model=attack_model,
                  target_model=target_model,
                  eval_model=eval_model,
                  jailbreak_datasets=dataset)

  # Finally, start jailbreaking.
  attacker.attack(save_path = save_path, max_workers=max_workers)
elif attack_method == "CodeChameleon":
  attacker = CodeChameleon(attack_model=attack_model,
                  target_model=target_model,
                  eval_model=eval_model,
                  jailbreak_datasets=dataset)
  
  attacker.attack(save_path = save_path, max_workers=max_workers)
elif attack_method == "DeepInception":
  attacker = DeepInception(attack_model=attack_model,
                  target_model=target_model,
                  eval_model=eval_model,
                  jailbreak_datasets=dataset)
  
  attacker.attack(save_path = save_path, max_workers=max_workers)
elif attack_method == "Chipher":
  attacker = Cipher(attack_model=attack_model,
                  target_model=target_model,
                  eval_model=eval_model,
                  jailbreak_datasets=dataset)
  
  attacker.attack(save_path = save_path, max_workers=max_workers)