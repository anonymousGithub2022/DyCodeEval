import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import HumanEvalData, MBPPData, SyntheticData
from src.codellm import AbstLLM, LLaMaLLM, CodeDeepSeekLLM, Claude3_5LLM

SPLIT_SYM = "::::"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OVERFIT_DIR = os.path.join(RESULTS_DIR, "overfit_dir")
os.makedirs(OVERFIT_DIR, exist_ok=True)
GENERATED_CODE_DIR = os.path.join(RESULTS_DIR, "generated_code")
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
FINAL_RES = "final_res"
os.makedirs(FINAL_RES, exist_ok=True)
EXE_ENV_DIR = "exe_env"
os.makedirs(EXE_ENV_DIR, exist_ok=True)
NEW_PROMPT_DIR = os.path.join(RESULTS_DIR, "new_prompt")
os.makedirs(NEW_PROMPT_DIR, exist_ok=True)
EXE_RES_DIR = os.path.join(RESULTS_DIR, "exe_res_dir")
os.makedirs(EXE_RES_DIR, exist_ok=True)
PASS_AT_K_DIR = os.path.join(RESULTS_DIR, "pass_at_k")
os.makedirs(PASS_AT_K_DIR, exist_ok=True)

PARTIAL_LIST = [
    0, 0.25, 0.5, 0.75, 1.0
]

def model_id2name_cls(model_id: int):
    if model_id == 0:
        model_name = "meta-llama/Llama-3.2-1B"
        model_cls = LLaMaLLM
        is_lora = True
    elif model_id == 1:
        model_name = "meta-llama/Llama-3.2-3B"
        model_cls = LLaMaLLM
        is_lora = True
    elif model_id == 2:
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        model_cls = CodeDeepSeekLLM
        is_lora = True
    elif model_id == 3:
        model_name = "meta-llama/Llama-3.1-8B"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 4:
        model_name = "meta-llama/CodeLlama-7b-hf"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 5:
        model_name = "meta-llama/CodeLlama-13b-hf"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 6:
        model_name = "deepseek-ai/DeepSeek-V2-Lite"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 7:
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 8:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 9:
        model_name = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        model_cls = Claude3_5LLM
        is_lora = False
    elif model_id == 10:
        model_name = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        model_cls = Claude3_5LLM
        is_lora = False
    elif model_id == 11:
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 12:
        model_name = "Qwen/Qwen2.5-Coder-7B"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 13:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 14:
        model_name = "Qwen/Qwen2.5-7B"
        model_cls = LLaMaLLM
        is_lora = False
    else:
        raise ValueError(f"Model ID {model_id} is not valid")
    return model_name, model_cls, is_lora


def load_finetune_model(model_id: int):
    if model_id in [0, 1, 2]:
        model_name, _, _ = model_id2name_cls(model_id)
    else:
        raise ValueError(f"Model {model_id} not supported")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.model_name = model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_benchmark_model(model_id: int) -> AbstLLM:
    model_name, model_cls, is_lora = model_id2name_cls(model_id)
    model = model_cls(model_name, is_lora)

    model.model_name = model_name.split('/')[-1]
    return model


def load_benchmark_model_name(model_id: int) -> str:
    model_name, model_cls, is_lora = model_id2name_cls(model_id)

    return model_name.split('/')[-1]


def make_task_name(model_name, dataset, partial):
    if partial is not None:
        task_name = model_name + SPLIT_SYM + dataset.data_name + SPLIT_SYM + str(partial)
    else:
        assert dataset is None
        task_name = model_name
    return task_name


def load_my_dataset(data_id: int):
    if data_id == 0:
        dataset = HumanEvalData()
    elif data_id == 1:
        dataset = MBPPData()
    elif data_id == 2:
        save_dir = os.path.join(
            NEW_PROMPT_DIR,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            'filted_HumanEval')
        dataset = SyntheticData(save_dir, "claude-3-5-haiku")
    elif data_id == 3:
        save_dir = os.path.join(
            NEW_PROMPT_DIR,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            'filted_MBPP')
        dataset = SyntheticData(save_dir, "claude-3-5-haiku")
    elif data_id == 4:
        save_dir = os.path.join(
            NEW_PROMPT_DIR,
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            'filted_HumanEval')
        dataset = SyntheticData(save_dir, "claude-3-5-sonnet")
    elif data_id == 5:
        save_dir = os.path.join(
            NEW_PROMPT_DIR,
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            'filted_MBPP')
        dataset = SyntheticData(save_dir, "claude-3-5-sonnet")
    else:
        raise NotImplementedError
    return dataset


def load_lora(model_name, dataset, partial):
    task_name = make_task_name(model_name, dataset, partial)
    output_dir = os.path.join(OVERFIT_DIR, task_name)
    if not os.path.exists(output_dir):
        return None
    sub_dirs = sorted(list(os.listdir(output_dir)))
    return os.path.join(output_dir, sub_dirs[-1])


if __name__ == '__main__':
    load_my_dataset(2)
