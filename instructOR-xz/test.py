import json
import tqdm

def download_models():
    import os

    os.environ['http_proxy'] = 'socks5h://127.0.0.1:1080'
    os.environ['https_proxy'] = 'socks5h://127.0.0.1:1080'

    from transformers import AutoTokenizer, AutoModel
    import torch
    # Sentences we want sentence embeddings for
    sentences = ["样例数据-1", "样例数据-2"]

    # Load model from HuggingFace Hub
    # model_id = "BAAI/bge-large-zh"
    model_id = "BAAI/bge-large-zh-noinstruct"
    # model_id= "BAAI/bge-base-zh"
    # model_id= "BAAI/bge-small-zh"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    modelpath = os.path.join(os.environ['HOME'], "CommonModels", model_id)

    tokenizer.save_pretrained(modelpath)
    model.save_pretrained(modelpath)
    
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("Sentence embeddings:", sentence_embeddings)

def testModelEmb():
    import os

    from torch import nn
    from embedder import ContrastiveEmbedder
    from transformers import AutoTokenizer
    modelpath = os.path.join(os.environ['HOME'], "SharedData/instructOR-encoders")
    # modelpath = os.path.join(os.environ['HOME'], "CommonModels/moka-ai/m3e-base")

    model = ContrastiveEmbedder.from_pretrained(modelpath, pooling_strategy="last_mean")
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    print(model)

    texts = ["我不喜欢苹果", "我喜欢苹果", "我喜欢水果"]
    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # output = model(**inputs)
    output = model.batch_encode(texts, tokenizer, 32, 512, return_tenor=True)
    print(output)
    output= output[0]

    print(output.size())
    print(output)

    similarity = nn.CosineSimilarity(dim=-1)

    sim = similarity(output.unsqueeze(1), output.unsqueeze(0))

    print(sim)


def testdataloader():
    import os
    from dataloader import load_dataset_local
    dataPath = os.path.join(os.environ['HOME'], 'SharedData/instructOR-MEDI')
    ds = load_dataset_local("MEDI")['train']

    print(ds)

    tasks = set(ds['task_name'])

    print(tasks)
    dslist = []
    for task in tasks:
        dslist.append(ds.filter(lambda x: x['task_name']== task, num_proc=32))
        dslist[-1].to_json(
            os.path.join(dataPath, task+".jsonl"), 
            force_ascii=False,
            num_proc=max(1,min(len(ds)//100000, 32))
        )
        print(f"saved {task}/(size={len(dslist[-1])})")
        
    print(len(dslist), len(tasks))


def load_json(kv):
    fn, file = kv
    records = []
    with open(file) as f:
        for line in tqdm.tqdm(f):
            x= json.loads(line)
            x= {
                "text": " ".join(x['query']),
                "text_pos": " ".join(x['pos']),
                "text_neg": " ".join(x['neg']),
            }
            records.append(x)
    return [fn, records]

def transform_medi_format():
    import os
    from datasets import Dataset
    from multiprocessing import Pool
    dataPath = os.path.join(os.environ['HOME'], 'SharedData/instructOR-MEDI')

    files = {fn: os.path.join(dataPath, "raws", fn) 
        for i, fn in enumerate(os.listdir(os.path.join(dataPath, "raws")) )
    }
    
    workers = Pool(32)

    fn_list = workers.map(load_json, files.items())
    workers.close()
    workers.join()

    for fn, records in fn_list:
        ds = Dataset.from_list(records)
        
        ds.select_columns(["text", "text_pos"]).to_json(
            os.path.join(dataPath, "pairs", fn+".jsonl"), 
            force_ascii=False,
            num_proc=max(1,min(len(records)//100000, 32))
        )
        ds.select_columns(["text", "text_pos", "text_neg"]).to_json(
            os.path.join(dataPath, "triplets", fn+".jsonl"), 
            force_ascii=False,
            num_proc=max(1,min(len(records)//100000, 32))
        )
        print(f"saved {fn}/(size={len(records)})")
        


def test_multi_task_dataset():
    import os
    import tqdm
    import random
    from datasets import load_dataset
    from dataloader import MultiTaskDataset, MultiTaskDistributedSampler
    from embedder import MultiTaskTrainer
    datafmt = "pairs"
    base_str = ([chr(ord('a')+i) for i in range(26)] 
                    #    +[chr(ord('A')+i) for i in range(26)]
                       )
    random.shuffle(base_str)
    def get_chr_r(num):
        res_str = ""
        if num ==0:
            return base_str[0]
        
        while num>0:
            idx = num% len(base_str)
            num= num// len(base_str)
            res_str+= base_str[idx]
        return res_str
    
    dataPath = os.path.join(os.environ['HOME'], 
            'SharedData/instructOR-MEDI/'+ datafmt
    )
    files = {get_chr_r(i): os.path.join(dataPath, fn) 
             for i, fn in enumerate(os.listdir(dataPath)[:30] ) }
    print(len(files))
    dataset = load_dataset("json", data_files=files, num_proc=32)

    print(dataset)

    dslist = list(dataset.values())
    mds =MultiTaskDataset(dslist, rank=0)

    print(len(mds.datasets_full))
    print(len(mds))

    mds =MultiTaskDataset(dslist, rank = 0, world_size=2)

    print(len(mds))

    for i in range(0):
        mds.reset_epoch()
        for x in tqdm.tqdm(mds):
            ...
            # i+=1
            # if i%1000==0:
            #     print(x)
        


    from dataloader import TripletCollator, PairCollator
    from transformers import AutoModel, AutoTokenizer, TrainingArguments
    args = TrainingArguments(
        output_dir="test123",
        per_device_train_batch_size=32,
        # per_gpu_train_batch_size=32  #deprecated args
    )
    model = AutoModel.from_pretrained(
        os.path.join(os.environ['HOME'], "CommonModels/moka-ai/m3e-small/")
    )
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
    if datafmt== "triplets":
        data_collator = TripletCollator(tokenizer, pad_to_multiple_of=8, 
                        return_tensors="pt", padding=True, max_length=512)
    else:
        data_collator = PairCollator(tokenizer, pad_to_multiple_of=8, 
                        return_tensors="pt", padding=True, max_length=512)
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer= tokenizer,
        train_dataset=mds, 
        args=args,
        data_collator=data_collator
    )

    dataloader = trainer.get_train_dataloader()
    print(dataloader.dataset)
    print(dataloader.dataset[0])

    count =0
    for i in range(3):
        for x in tqdm.tqdm(dataloader):
            ...
            count +=1
            if count % 10 == 0:
                print({x0: x[x0].size() for x0 in x })


download_models()

# transform_medi_format()
# testdataloader()
# test_multi_task_dataset()
    