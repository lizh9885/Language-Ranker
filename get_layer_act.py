import os
import torch
# from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
import accelerate
import pickle
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_dataset,get_llama_activations_bau
model_map_dict = {
    "llama-7b":"./model/llama-2-7b-hf",
    "gemma-7b":"./model/gemma-7b",
    "qwen-7b":"./model/qwen-7b",
    "mistral-7b":"./model/mistral-7b/Xorbits/Mistral-7B-v0___1",
    "qwen1.5-0.5b":"./model/qwen1.5-0.5b",
    "qwen1.5-1.8b":"./model/qwen1.5-1.8b",
    "qwen2-7b":"../model_ckpt/qwen2-7b",
    "qwen1.5-4b":"./model/qwen1.5-4b",
    "qwen1.5-7b":"./model/qwen1.5-7b",
    "llama3-8b":"../model_ckpt/llama3-8b"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2-7b')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/opus-100')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()
    
    MODEL = model_map_dict[args.model_name] if not args.model_dir else args.model_dir
    print(MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(MODEL,device_map="auto",torch_dtype=torch.float16, trust_remote_code=True)
    # for name, param in llm_model.named_parameters():  
    #     print(name,param.size())  
    
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    device = "cuda"
    
    # 创建数据集
    data_json_dict = format_dataset(args.dataset_dir,llm_tokenizer)

    
    for lang_pair, lang_pair_list in data_json_dict.items():
        all_layer_wise_activations = []
        all_head_wise_activations = []
        for single_pair_id in lang_pair_list: #每对句子
            lang1_sen = single_pair_id[0]
            lang2_sen = single_pair_id[1]
            id_1 = llm_tokenizer(lang1_sen, return_tensors = 'pt').input_ids
            id_2 = llm_tokenizer(lang2_sen, return_tensors = "pt").input_ids
            layer_wise_activations_1 = get_llama_activations_bau(llm_model, id_1, device)
            all_layer_wise_activations.append(layer_wise_activations_1[:,-1,:])#(32,1,hidden_size)
            # all_head_wise_activations.append(head_wise_activations_1[:,-1,:])
            layer_wise_activations_2= get_llama_activations_bau(llm_model, id_2, device)
            all_layer_wise_activations.append(layer_wise_activations_2[:,-1,:])#(32,1,hidden_size)
            # all_head_wise_activations.append(head_wise_activations_2[:,-1,:])

        print("Saving layer wise activations")
        np.save(f'middle_result/{args.model_name}/{lang_pair}_layer_wise.npy', all_layer_wise_activations)
        
        # # print("Saving head wise activations")
        # np.save(f'middle_result/{args.model_name}/{lang_pair}_head_wise.npy', all_head_wise_activations)



if __name__ == "__main__":
    main()