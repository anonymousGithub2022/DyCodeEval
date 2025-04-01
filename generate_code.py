import os
import argparse
import torch
import json

from src.data import MyDataset, CodeTask
from src.codellm import CodeLLaMaLLM, AbstLLM
from copy import deepcopy

from utils import load_benchmark_model
from utils import load_my_dataset
from utils import PARTIAL_LIST, load_lora, make_task_name
from utils import GENERATED_CODE_DIR, SPLIT_SYM


def generate(code_llm, eval_dataset, sample_num, save_dir, override):

    new_eval_dataset = []
    grouped_data = eval_dataset.group()
    for k in grouped_data.keys():
        new_eval_dataset.extend(grouped_data[k][:sample_num])

    for i in range(sample_num):
        task_dir = os.path.join(save_dir, str(i))
        os.makedirs(task_dir, exist_ok=True)

    final_eval_dataset = []
    for data in new_eval_dataset:
        for i in range(sample_num):
            tmp_data = deepcopy(data)
            file_id = str(tmp_data.data_id).replace('/', '-')
            task_dir = os.path.join(save_dir, str(i))
            save_path = os.path.join(task_dir, f"code_{file_id}.py")

            tmp_data.save_path = save_path
            tmp_data.task_dir = task_dir
            final_eval_dataset.append(tmp_data)

    res = code_llm.code_gen_batch(final_eval_dataset)
    raw_res_path = os.path.join(save_dir, 'raw_res.tar')
    if override or not os.path.exists(raw_res_path):
        torch.save(res, raw_res_path)
        print(f'save meta data to {raw_res_path}')

    for output in res:
        save_file = output.original_task.save_path

        import_st = output.original_task.import_str
        test_cases = output.original_task.test_case_str
        final_code = import_st + '\n\n\n' + output.final_code + '\n\n\n' + test_cases

        if override or not os.path.exists(save_file):
            with open(save_file, 'w') as f:
                f.write(final_code)


def main(args):
    sample_num = args.n
    if args.temperature == 0:
        sample_num = 1
    lora_dataset = load_my_dataset(args.lora_data_id)

    code_llm = load_benchmark_model(args.model_id)

    partial = PARTIAL_LIST[args.partial_id]
    lora_path = load_lora(code_llm.model_name, lora_dataset, partial)
    if lora_path is not None:
        task_name = make_task_name(code_llm.model_name, lora_dataset, partial)
    else:
        task_name = make_task_name(code_llm.model_name, None, None)

    print(f"model name is {task_name}")
    config = {
        'temperature':  args.temperature,
        "top_p": args.top_p,
        "max_tokens": 1024,  # args.max_tokens,
        "tp_size": 1,  # args.tp_size,
        "dtype": "float16",
        'lora_path': lora_path,
        "stop": [
                "\n>>>", "\n$", '\nclass',
                '\ndef', '\n#', '\nprint',
                 "\n@",
                "\nif __name__ == '__main__':",
                '\nif __name__ == "__main__":'

            ]

    }
    code_llm.init_ai_kwargs(config)

    eval_dataset = load_my_dataset(args.data_id)
    save_dir = os.path.join(
        GENERATED_CODE_DIR, task_name, eval_dataset.data_name,
        f"temp_{args.temperature}{SPLIT_SYM}top_p{args.top_p}"
    )
    os.makedirs(save_dir, exist_ok=True)
    if args.override or not os.path.join(save_dir, 'config.json'):
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
    generate(code_llm, eval_dataset, sample_num, save_dir, args.override)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--model_id', type=int, default=10)
    parser.add_argument('--lora_data_id', type=int, default=1)
    parser.add_argument('--partial_id', type=int, default=4)
    parser.add_argument('--data_id', type=int, default=1)
    parser.add_argument('--override', type=bool, default=True)

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()
    for model_id in [6, 7]:
        args.model_id = model_id
        for data_id in range(4):
            args.data_id = data_id
            main(args)
