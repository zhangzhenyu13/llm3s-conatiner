import os
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset

basePath = os.path.join(os.environ['HOME'], "SharedData/instructOR-datasets")

def load_one(dataset_id, data_format):
    dataPath = os.path.join(basePath, data_format, dataset_id, "train.jsonl")

    files = {"train": dataPath}
    ds = load_dataset("json", data_files=files)['train']
    if len(ds)> maxSamples:
        indices = np.arange(len(ds))
        np.random.shuffle(indices)
        indices= indices[:maxSamples]
        ds = ds.select(indices=indices)
    
    taskid = dataset_id.replace("/", "-")
    ds.select_columns(["text", "text_pos"]).to_json(
            os.path.join(basePath, "sampled-pairsI", taskid+".jsonl"), 
            force_ascii=False,
            num_proc=max(1,min(len(ds)//100000, 32))
        )
    print(f"saved {taskid}/(size={len(ds)})")
    # return ds

def sample_func(taskid):
    try:
        load_one(taskid, dataFMT)
        print("finished:", taskid)
    except FileNotFoundError as e:
        print(e)
        # raise e
    except Exception as e:
        print(e, "errors:", taskid)
        raise e

if __name__ == "__main__":
    from multiprocessing import Pool
    from instructions_emb import instructionsMapper
    dataFMT= "pairsI"
    maxSamples = int(5e5)
    tasklist = instructionsMapper.keys()
    # tasklist =['mlqa/mlqa-translate-train.zh']
    workers = Pool(32)
    workers.map(sample_func, tasklist)

    workers.close()
    workers.join()

        
        

