import json
import os
import numpy as np
import argparse
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
from utils import format_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

model_map_dict = {
    "llama-7b":"../model/llama-2-7b-hf",
    "gemma-7b":"../model/gemma-7b",
    "qwen-7b":"../model/qwen-7b",
    "mistral-7b":"../model/mistral-7b/Xorbits/Mistral-7B-v0___1",
    "qwen1.5-0.5b":"./model/qwen1.5-0.5b",
    "qwen1.5-1.8b":"./model/qwen1.5-1.8b",
    "qwen2-7b":"../model_ckpt/qwen2-7b",
    "qwen1.5-4b":"./model/qwen1.5-4b",
    "qwen1.5-7b":"./model/qwen1.5-7b",
    "llama3-8b":"../model_ckpt/llama3-8b"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type",type=str,default='arc')
    parser.add_argument('--model_name', type=str, default='mistral-7b')
    # parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()
    
    MODEL = model_map_dict[args.model_name] if not args.model_dir else args.model_dir
    # print(MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(MODEL,device_map="auto",torch_dtype=torch.float16, trust_remote_code=True)
    # for name, param in llm_model.named_parameters():  
    #     print(name,param.size())  
    
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    device = "cuda"

    test_language =  ["zh","de","fr","es","it","kn","hi","hy","mr","te"]

    for l in test_language:
        #每一类测试子集
        all_data  = format_dataset(args.task_type,l)
        #构造demo
        random_demo = random.sample(all_data,4)

        
        # 每个sample
        acc = 0
        for single_test in all_data:
            prompt_parts = [f"Question: {item['instruction']} A: {item['option_a']} B: {item['option_b']} C: {item['option_c']} D: {item['option_d']}\nAnswer:{item['answer']}" for item in random_demo]
            prompt_parts.append(f"Question: {single_test['instruction']} A: {single_test['option_a']} B: {single_test['option_b']} C: {single_test['option_c']} D: {single_test['option_d']}\nAnswer:")
            prompt = "\n\n".join(prompt_parts)
            
            model_input = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
            
            with torch.no_grad():
                output = llm_model.generate(**model_input, max_new_tokens=2)[0]
        
                response = llm_tokenizer.decode(output[len(model_input["input_ids"][0]):], skip_special_tokens=True)
                
                if single_test["answer"] in response:
                    acc+=1
        print(f"MODEL:{args.model_name}, DATASET:{args.task_type}, LANGUAGE:{l}, length:{len(all_data)}, correct:{acc},acc{acc/len(all_data)}")


if __name__ == "__main__":
    
    main()