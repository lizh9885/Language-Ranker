import json
import os
import random

def format_dataset(task_type="arc",language = "de"):
    if task_type == "arc":
        with open(f"./datasets/datasets/m_arc/{language}_sample_test.json","r",encoding="utf8") as fr:
            all_data =  json.load(fr)
        # filter_all_data = [item for item in all_data if len(item)==7]
        # if len(filter_all_data) >= 100:
        #     filter_all_data = random.sample(filter_all_data,100)
        
    if task_type == "mmlu":
        with open(f"./datasets/datasets/m_mmlu/{language}_sample_test.json","r",encoding="utf8") as fr:
            all_data =  json.load(fr)
        # filter_all_data = [item for item in all_data if len(item)==7]
        # if len(filter_all_data) >= 100:
        #     filter_all_data = random.sample(filter_all_data,100)
        
    return all_data
    
test_language =  ["zh","de","fr","es","it","kn","hi","hy","mr","te"]
for language in test_language:
    with open(f"./datasets/datasets/m_arc/{language}_test.json","r",encoding="utf8") as fr:
        all_data =  json.load(fr)
        filter_all_data = [item for item in all_data if len(item)==7]
        if len(filter_all_data) >= 100:
            filter_all_data = random.sample(filter_all_data,100)
    with open(f"./datasets/datasets/m_arc/{language}_sample_test.json","w",encoding="utf8") as fw:
        fw.write(json.dumps(filter_all_data,ensure_ascii=False,indent=2))
for language in test_language:
    with open(f"./datasets/datasets/m_mmlu/{language}_test.json","r",encoding="utf8") as fr:
        all_data =  json.load(fr)
        filter_all_data = [item for item in all_data if len(item)==7]
        if len(filter_all_data) >= 100:
            filter_all_data = random.sample(filter_all_data,100)
    with open(f"./datasets/datasets/m_mmlu/{language}_sample_test.json","w",encoding="utf8") as fw:
        fw.write(json.dumps(filter_all_data,ensure_ascii=False,indent=2))
