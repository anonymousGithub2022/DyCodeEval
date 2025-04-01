import os
import json
import random
import numpy as np
from utils import EXE_RES_DIR
from post import extract_meta
import csv
from eval_pass_K import estimator


final_csv_res = {}
for file_name in os.listdir(EXE_RES_DIR):
    file_path = os.path.join(EXE_RES_DIR, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    if len(data) == 0:
        continue
    model_name, data_name, hyper = extract_meta(file_name)
    if "::::" in model_name:
        continue
    if data_name in ["MBPP" , "HumanEval"]:
        continue
    grouped = {}
    for k in data.keys():
        p_id = k.split('____SPLIT____')[0]
        if p_id not in grouped:
            grouped[p_id] = []
        grouped[p_id].append(data[k])
    if max([len(d) for d in grouped.values()]) == 1:
         continue
    random_pass_1 = []
    for _ in range(5):
        all_res = []
        for p_id in grouped.keys():
            sample = random.choice(grouped[p_id])
            n = len(sample)
            c = len([d for d in sample.values() if d is True])

            pass_1 =  estimator(n, c, 1)
            all_res.append(pass_1)
        random_pass_1.append(np.mean(all_res))
    if data_name not in final_csv_res:
        final_csv_res [data_name] = []
    final_csv_res[data_name].append([model_name] + random_pass_1)
for data_name in final_csv_res:
    data = final_csv_res[data_name]
    sorted_data = sorted(data, key=lambda sublist: sum(sublist[1:]) / len(sublist[1:]) if sublist else float('inf'))

    with open(f"final_res/{data_name}_stability.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        for row in sorted_data:
            writer.writerow(row)
