import json
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen-7b')
model_map_dict = {
    "llama-7b":"./middle_result/llama-7b/",
    "gemma-7b":"./middle_result/gemma-7b/",
    "mistral-7b":"./middle_result/mistral-7b/",
    "qwen-7b":"./middle_result/qwen-7b/",
    "qwen1.5-0.5b":"./middle_result/qwen1.5-0.5b",
    "qwen1.5-1.8b":"./middle_result/qwen1.5-1.8b",
    "qwen2-7b":"./middle_result/qwen2-7b",
    "qwen1.5-4b":"./middle_result/qwen1.5-4b",
    "qwen1.5-7b":"./middle_result/qwen1.5-7b"
}
args = parser.parse_args()
result_dir = model_map_dict[args.model_name]
print_list = []
for root, dirs, files in os.walk(result_dir):  
    for file in files:  
        if "layer" in file: 
            file_path = os.path.join(root, file)  
            lang_pair_name = file.split("_")[0]
            lang1 = lang_pair_name.split("-")[0]
            lang2 = lang_pair_name.split("-")[1]
            en_index = -1
            if "en" in lang1:
                en_index = 0
            else:
                en_index = 1
            data = np.load(file_path) #[4000个(hidden_layers, hidden_dim)]
            assert len(data)%2 == 0
            print(len(data))
            hidden_layers = data[0].shape[0]
            
            
            sim_list = [0 for _ in range(hidden_layers)]
            for i in range(0, len(data), 2):
                lang1_vec = data[i] #(hidden_layers, hidden_dim)
                lang2_vec = data[i+1] #(hidden_layers, hidden_dim)
                for layer in range(hidden_layers):
                    lang1_layer_vec = lang1_vec[layer]
                    lang2_layer_vec = lang2_vec[layer]
                    
                    lang1_layer_vec = lang1_layer_vec
                    lang2_layer_vec = lang2_layer_vec
                    norm_v1 = np.linalg.norm(lang1_layer_vec)  
                    norm_v2 = np.linalg.norm(lang2_layer_vec)  
                    
                    cosine_similarity = np.dot(lang1_layer_vec/norm_v1, lang2_layer_vec/norm_v2)
                    if np.isnan(cosine_similarity):  
                        print(lang1_layer_vec, lang2_layer_vec)  
                        print(np.dot(lang1_layer_vec, lang2_layer_vec))
                    sim_list[layer] += cosine_similarity
            
            h = list(np.array(sim_list)/(len(data)//2))
            print(h)

            print_dict = {
                "lang_name": lang_pair_name,
                "res":h
            }
            print_list.append(print_dict)
            
final_dir = os.path.join("./result/",f"{args.model_name}_new.json")
with open(final_dir,"w",encoding="utf8") as fw:
    for dictionary in print_list:  
        json.dump(dictionary, fw)  
        fw.write('\n')  



