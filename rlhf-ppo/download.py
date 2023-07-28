# download dataset to local machine
import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
import tqdm, json

# if you have proxy running in your server, you can uncomment below to boost speed
# os.environ['http_proxy'] = 'socks5h://127.0.0.1:1080'
# os.environ['https_proxy'] = 'socks5h://127.0.0.1:1080'

dataset_list = [
    # "Dahoas/rm-static", "Dahoas/full-hh-rlhf", "Dahoas/synthetic-instruct-gptj-pairwise",
    # "yitingxie/rlhf-reward-datasets"
    # "openai/webgpt_comparisons", 
    # "stanfordnlp/SHP"
    # "tasksource/oasst1_pairwise_rlhf_reward"
    # "andersonbcdefg/dolly_reward_modeling_pairwise"
    # "andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai"
    # "Anthropic/hh-rlhf"
    "FreedomIntelligence/ShareGPT-CN"
]

# dataset_list = [
#     # "sunzeyeah/chinese_chatgpt_corpus"
#     # "wangrui6/Zhihu-KOL","Cohere/miracl-zh-queries-22-12","Hello-SimpleAI/HC3-Chinese",#"mkqa"
#     # "liyucheng/zhihu_rlhf_3k"
# ]

def data_cache_local():
    for dataset_id in dataset_list:
        print('dataset:', dataset_id )
        if "Hello" in dataset_id:
            dataset = load_dataset("hello-simpleai/hc3", data_files=['all.jsonl' ])
        else:
            dataset = load_dataset(dataset_id, ignore_verifications=True )
        print(dataset)
        print("=======")

        if isinstance(dataset, DatasetDict):
            data = DatasetDict()
            for key in dataset.keys():

                savePath = os.path.join(os.environ['HOME'], 'SharedData/RLHF/dataset', dataset_id, key)
                dataset[key].save_to_disk(savePath)
                print(key, savePath)
                ds =  load_from_disk(savePath)
                print(ds)
                data[key] = ds
            print(data)
            print("********")
        else:
            savePath = os.path.join(os.environ['HOME'], 'SharedData/RLHF', dataset_id, "data")
            dataset.save_to_disk(savePath)


data_cache_local()