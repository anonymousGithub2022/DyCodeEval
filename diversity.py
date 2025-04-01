import nltk
import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt_tab')
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import os
from utils import load_my_dataset
from copy import deepcopy

device = torch.device(1)
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


prompt_dir = "./final_res/prompt"

if not os.path.isdir(prompt_dir):
    os.mkdir(prompt_dir)



def compute_bleu(base_sentences, new_sentences):

    new_sentences = [nltk.word_tokenize(text.lower()) for text in new_sentences]
    base_sentences = [nltk.word_tokenize(text.lower()) for text in base_sentences]

    if len(new_sentences) != len(base_sentences):
        return np.nan

    n = len(new_sentences)
    total_bleu_score = 0.0

    for i in tqdm(range(n)):
        ref_sentences = [base_sentences[i]]
        hypothesis = new_sentences[i]
        bleu_score = sentence_bleu(ref_sentences, hypothesis)
        total_bleu_score += bleu_score

    self_bleu_score = total_bleu_score / (n + 1e-12)
    return self_bleu_score


def compute_perplexity(generated_texts):

    encodings = tokenizer("\n\n".join(generated_texts), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    if nlls == []:
        return np.nan
    ppl = torch.exp(torch.stack(nlls).mean())
    return float(ppl)


def compute_embeding_sim(base_sentences, new_sentences):
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    model = model.to(device).eval()
    vec1 = model.encode(base_sentences)
    vec2 = model.encode(new_sentences)
    if len(vec2) == 0:
        return 0.0
    cos_sim = cosine_similarity(vec1, vec2)
    sim = np.diagonal(cos_sim)
    return np.mean(sim)


def read_prompt(file_path):
    with open(file_path, 'r') as f:
        prompt = f.read()
    return prompt


def read_approach_prompt(dataset_name, approach):
    save_dir = f"./workdir/codegen/{dataset_name}/{approach}/codegen-2b_temp_0.7/"
    results = {}
    for dirpath, dirnames, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename == "prompt.txt":
                file_path = os.path.join(dirpath, filename)
                problem_id = int(dirpath.split('_')[-1])
                results[problem_id] = read_prompt(file_path)
    return results


def compute_diffimp(base_sentences, new_sentences):
    res = []
    for s1, s2 in zip(base_sentences, new_sentences):
        if s1 == s2:
            res.append(0)
        else:
            res.append(1)
    return np.array(res)

def check_implementation_difference(base_sentences, new_sentences):
    res = compute_diffimp(base_sentences, new_sentences)
    return res.mean()





def external_diversity():
    for data_id in range(2, 4):
        seed_data_id = data_id % 2
        new_data = load_my_dataset(data_id)
        new_data = new_data.group()
        seed_data = load_my_dataset(seed_data_id)
        seed_data = seed_data.group()

        base_sents = [seed_data[k][0].docstring for k in new_data]
        new_sents = [new_data[k][0].docstring for k in new_data]
        bleu = compute_bleu(base_sents, new_sents)
        # perplexity = compute_perplexity(new_sents)
        sim = compute_embeding_sim(base_sents, new_sents)
        print(data_id, bleu, sim)




def internal_diversity():
    for data_id in range(2, 4):

        new_data = load_my_dataset(data_id)
        new_data = new_data.group()

        base_sents = [new_data[key][0].docstring for key in new_data]
        new_sents = [new_data[key][1].docstring if len(new_data[key]) > 1 else new_data[key][0].docstring for key in new_data]
        bleu = compute_bleu(base_sents, new_sents)

        sim = compute_embeding_sim(base_sents, new_sents)
        print(data_id, bleu, sim)



def prompt2text():
    approach_list = [
        "base", "add_demo", "del_demo", "rep_demo",
        "char_mutation", "token_mutation",
        "func_name", "insert_line", "comment",
        "output_v_mutation", "output_mutation",
    ]
    random_prompt_dir = "random_prompt"
    prompt_text_dir = "prompt_text"
    if not os.path.isdir(prompt_text_dir):
        os.mkdir(prompt_text_dir)

    for dataset_name in ['humaneval', 'mbpp']:
        for approach in approach_list:
            task_name = f"{dataset_name}::::{approach}"
            save_path = os.path.join(random_prompt_dir, task_name)
            all_prompt = torch.load(save_path)
            prompt_1, prompt_2 = all_prompt
            save_path = os.path.join(prompt_text_dir, task_name + '.txt')
            with open(save_path, 'w') as f:
                for i, k in enumerate(sorted(prompt_1.keys())):
                    if i == 100:
                        break
                    prompt_str = prompt_1[k]
                    f.write(prompt_str)
                    f.write("    pass\n\n\n")






if __name__ == '__main__':
    external_diversity()
    print('--------------------')
    internal_diversity()
    # prompt2text()
    # diffimp_curve()