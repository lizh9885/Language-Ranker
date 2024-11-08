import os
import json
import torch

lang_name = ["de","fr","es","pt","nl","it","ro","cs",
             "pl","ar","he","tr","ja","ko","vi","th",
             "id","ms","km","hi","bn","ur","zh","ig","kk","kn","or","tk"]
def read_json(file):
    with open(file, "r", encoding="utf8") as fr:
        all_data = json.load(fr)
    return all_data
def format_dataset(dataset_dir, tokenizer):
    json_dict = {}
    for root, dirs, files in os.walk(dataset_dir):  
        for file in files:  
            if file.endswith('json') and "test" in file: 
        
                in_file = os.path.join(root, file)  
                parent_folder_name = os.path.basename(root)
                json_dict[parent_folder_name] = read_json(in_file) #{"en-zh":[...]}
                
            

    filter_json_dict = {}
    # post process 
    for key,value in json_dict.items():
        lang1,lang2 = key.split("-")[0],key.split("-")[1]
        
        if lang1 != "en" and lang2 != "en":
            continue
        cur_lang = lang1 if lang2 == "en" else lang2
        if cur_lang not in lang_name:
            continue
        id_list = []
        for single_value in value:
            lang1_sen = single_value[lang1]
            lang2_sen = single_value[lang2]
            # lang1_id = tokenizer(lang1_sen, return_tensors = 'pt').input_ids
            # lang2_id = tokenizer(lang2_sen, return_tensors = "pt").input_ids
            id_list.append((lang1_sen,lang2_sen))
        filter_json_dict[key] = id_list
    return filter_json_dict


def get_llama_activations_bau(model, prompt, device): 
    # HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    
    layes_num = model.config.num_hidden_layers
    heads_num = model.config.num_attention_heads
    with torch.no_grad():
        prompt = prompt.to(device)
        # with TraceDict(model, HEADS) as ret:
        output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states[1:] #(32*(batch=1,seq_len,hidden_size))
       
        assert len(hidden_states) == layes_num
        stack_hidden_states = torch.stack(hidden_states, dim = 0)
        stack_hidden_states = stack_hidden_states.squeeze(1).detach().to(torch.float).cpu().numpy()#(32,seq_len,hidden_size)
        print(stack_hidden_states.shape)
        
        # head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]#[(4,4096)]
        # head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()#(32,seq_len,hidden_size)
        # print(head_wise_hidden_states.shape)
    return stack_hidden_states