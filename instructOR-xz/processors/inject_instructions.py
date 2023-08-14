import functools
import os
from datasets import load_dataset, DatasetDict
from transform_datasets import (
    save_ds_as_json, ds_map_func, concate_ds_dict,
    load_dataset_local
)


def inject_pairs(x, insts):
    if len(insts) ==1:
        return {
            "text": insts[0]+": "+ x['s1'].strip(),
            "text_pos": insts[0]+": "+ x['s2'].strip()
        }
    else:
        try:
            return {
                "text": insts[0]+": "+ x['s1'].strip(),
                "text_pos": insts[1]+": "+ x['s2'].strip()
            }
        except:
            print(x)
            return None

def inject_triplets(x, insts):
    if len(insts) ==1:
        return {
            "text": insts[0]+": "+ x['s1'].strip(),
            "text_pos": insts[0]+": "+ x['s2'].strip(),
            "text_neg": insts[0]+": "+ x['s3'].strip()
        }
    else:
        return {
            "text": insts[0]+": "+ x['s1'].strip(),
            "text_pos": insts[1]+": "+ x['s2'].strip(),
            "text_neg": insts[1]+": "+ x['s3'].strip()
        }

def transform_one(dataset_id, data_format):
    assert data_format in ("triplets", "pairs")

    dataPath = os.path.join(os.environ['HOME'], "SharedData/instructOR-datasets", data_format, dataset_id, "train.jsonl")
    files = {"train": dataPath}
    ds = load_dataset("json", data_files=files)
    
    match data_format:
        case "triplets": 
            ds = ds.rename_columns({"text":"s1", "text_pos":"s2", "text_neg":"s3"})
            ds = ds_map_func(ds, 
                functools.partial(inject_triplets, insts=instructionsMapper[dataset_id])
            )
            ds = ds.select_columns(["text", "text_pos", "text_neg"])
        case "pairs": 
            ds = ds.rename_columns({"text":"s1", "text_pos":"s2"})
            ds = ds_map_func(ds, 
                functools.partial(inject_pairs, insts=instructionsMapper[dataset_id])
            )
            ds= ds.select_columns(["text", "text_pos"])
        case _:
            raise ValueError(f"unknown ds format:{data_format}")

    save_ds_as_json(ds, data_format+"I", dataset_id)


def load_csl_abt_title(dataset_id='neuclir/csl'):
    insts=instructionsMapper[dataset_id]
    def wrapper_cate(x, text, text_pos):
        cate_desc= x['discipline']
        text = insts[0].replace("{CATE}", cate_desc)+": "+ text
        text_pos = insts[1].replace("{CATE}", cate_desc)+": "+ text_pos
        return {"text": text, "text_pos": text_pos}
    def kw_mapper(x):
        keywords = x['keywords']
        text = x['abstract']
        keywords = ", ".join(keywords)
        return wrapper_cate(x, keywords, text)
        
    def title_mapper(x):
        title = x['title']
        text = x['abstract']
        return wrapper_cate(x, title, text)
        
    
    dataset_dict = load_dataset_local(dataset_id)
    
    dataset_dict1 = ds_map_func(dataset_dict, kw_mapper
            ).select_columns(["text", "text_pos"])
    dataset_dict2 =  ds_map_func(dataset_dict, title_mapper
            ).select_columns(["text", "text_pos"])

    dataset_dict = concate_ds_dict([dataset_dict1, dataset_dict2])
    
    dataset_dict= DatasetDict(train= dataset_dict['csl'])
    # dataset_dict= dataset_dict.rename_columns({"text":"s1", "text_pos":"s2"})
    # dataset_dict = ds_map_func(dataset_dict, functools.partial(
    #     inject_pairs, insts=instructionsMapper[dataset_id]
    # ))

    
    return dataset_dict

def run_paris():
    dataFMT="pairs"
    begin=False
    for task in instructionsMapper:
        if task == "clue/chid":
            begin=True
        if not begin:
            continue
        if task in ["clue/chid", ]:
            continue

        print(f"\n*****{task}****\n")
        transform_one(task, dataFMT)

if __name__ == "__main__":
    from instructions_emb import instructionsMapper
    
    # run_paris()

    dsid='neuclir/csl'
    ds= load_csl_abt_title()
    save_ds_as_json(ds, "pairsI", dsid)
