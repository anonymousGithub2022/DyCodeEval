import os
import torch

from utils import FINAL_RES, NEW_PROMPT_DIR
from prompt_generation import LLM_LIST
from utils import load_my_dataset

import csv


def save_list_to_csv(data, filename):
    """Saves a list of lists to a CSV file, with each inner list as a row."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerows(data)  # Write each inner list as a row



for data_id in [2, 3, 4, 5]:
    data = load_my_dataset(data_id)
    res = data.group()
    seed_d_id = data_id % 2
    seed_data = load_my_dataset(seed_d_id)
    seed_data = seed_data.group()
    all_res = [['original', "   ", "GEN 1", "GEN 2", "GEN 3", "GEN 4" ,"GEN 5"]]
    for k in res:
        prefix = [seed_data[k][0]['prefix'],  "   "] + [d['prefix'] for d in res[k]]
        all_res.append(prefix)
    save_path = os.path.join(FINAL_RES, f"prompt_{data_id}.csv")
    save_list_to_csv(all_res, save_path)
