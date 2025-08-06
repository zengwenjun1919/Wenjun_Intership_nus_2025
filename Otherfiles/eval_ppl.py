import json
import random
import numpy as np
import torch
import os
import secrets
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from evaluate import load

# 是否开启绘图
plot_enabled = True  

# 读取 JSON 文件
def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

# 随机选取指定数量的样本
def get_samples(num, path):
    data = read_json(path)
    data = data[:num]
    text_samples = []
    for i in range(len(data)):
        if data[i]['input'] != '':
            text = data[i]['instruction'] + "\n" + data[i]['input']
        else:
            text = data[i]['instruction']
        # print(text)
        text_samples.append(text)
    return text_samples

# 计算 PPL
def compute_ppl(predictions):
    perplexity_metric = load("perplexity", module_type="metric")
    # print(predictions)
    results = perplexity_metric.compute(predictions=predictions, add_start_token=False, model_id='gpt2',max_length=1024)
    return results["perplexities"], results["mean_perplexity"]

if __name__ == '__main__':
    # 读取不同方法的数据
    dataset_paths = {
        "IF-DialogueTemplate": "/work/xzh/Concept-Fingerprint/data/if/if_chat_fp.json",
        "IF-SimpleTemplate": "/work/xzh/Concept-Fingerprint/data/if_fingerprint_mix.json",
        "UTF-LLAMA2": "/work/xzh/Concept-Fingerprint/data/utf/llama2_utf_fp.json",
        "UTF-MISTRAL": "/work/xzh/Concept-Fingerprint/data/utf/mistral-fp.json",
        "HashChain": "/work/xzh/Concept-Fingerprint/data/hash_chain/hc_fp.json",
        "Proflingo": "/work/xzh/Concept-Fingerprint/data/proflingo/proflingo_llama2.json",
        "XSUM" : "/work/xzh/Concept-Fingerprint/data/EverTracer/xsum.json",
        "AGNEWS":"/work/xzh/Concept-Fingerprint/data/EverTracer/agnews.json",
        "Alpaca_EN": "/work/xzh/comman_scripts/datasets/alpaca_data_52k.json",
        "Dolly_EN": "/work/xzh/comman_scripts/datasets/dolly_en_15k.json"
    }
    sample_sizes = {
        "IF-SimpleTemplate": 10,
        "IF-DialogueTemplate":8,
        "UTF-LLAMA2": 32,
        "UTF-MISTRAL": 32,
        "HashChain": 10,
        "Proflingo": 50,
        "Alpaca_EN": 500,
        "Dolly_EN": 500,
        "XSUM":100,
        "AGNEWS":100,
    }

    # 计算 PPL
    ppl_results = {}
    for method, path in dataset_paths.items():
        samples = get_samples(sample_sizes[method], path)
        ppl_values, mean_ppl = compute_ppl(samples)
        ppl_results[method] = {
            "ppl_values": np.array(ppl_values),
            "mean_ppl": mean_ppl
        }
        print(f"Mean Perplexity for {method}: {mean_ppl:.2f}")

    # 计算阈值 (基于 Dolly_EN 和 Alpaca_EN 的均值)
    base_mean_ppl = (ppl_results["Dolly_EN"]["mean_ppl"] + ppl_results["Alpaca_EN"]["mean_ppl"]) / 2
    threshold = base_mean_ppl * 1.5
    print(f"\nThreshold for filtering: {threshold:.2f}")

    # 过滤超过阈值的数据，并计算保留比例
    retained_ratios = {}
    for method, result in ppl_results.items():
        ppl_values = result["ppl_values"]
        retained_count = np.sum(ppl_values <= threshold)
        total_count = len(ppl_values)
        retained_ratio = retained_count / total_count
        retained_ratios[method] = retained_ratio
        print(f"{method}: Retained {retained_count}/{total_count} ({retained_ratio:.2%})")

    # 绘制 PPL 直方图（如果启用）
    if plot_enabled:
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(dataset_paths.keys()):
            plt.hist(np.log(ppl_results[method]["ppl_values"]), bins=20, edgecolor=colors[i], 
                     color=colors[i], alpha=0.5, label=method)

        plt.axvline(np.log(threshold), color='black', linestyle='--', label="Threshold")
        plt.xlabel('Log(Perplexity)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'PPL.png'), bbox_inches='tight', dpi=300)
        plt.show()


#CUDA_VISIBLE_DEVICES=0 python eval_ppl.py 