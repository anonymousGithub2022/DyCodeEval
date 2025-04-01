from datasets import load_dataset
import numpy as np
from utils import load_my_dataset
from src import CodeTask


lcb_codegen = load_dataset(
    "livecodebench/code_generation_lite", version_tag="v4_v5")
dataset = lcb_codegen['test'].to_list()
token_num_list = []
for data in dataset:
    token_num_list.append(len(data['question_content'].split(' ')))
print(min(token_num_list), np.mean(token_num_list), np.max(token_num_list))

for data_id in range(2):
    dataset = load_my_dataset(data_id)
    dataset = dataset.to_list()
    dataset = [CodeTask.from_dict(d).init() for d in dataset]
    token_num_list = []
    for data in dataset:
        token_num_list.append(len(data.instruction.split(' ')))
    print(min(token_num_list), np.mean(token_num_list), np.max(token_num_list))
