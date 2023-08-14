import os
from datasets import load_dataset

from transform_datasets import save_ds_as_json, ds_map_func

def transform_one(dataset_id):
    if not isinstance(dataset_id, str):
        dataset_id = "/".join(dataset_id)
    dataPath = os.path.join(os.environ['HOME'], "SharedData/instructOR-datasets/triplets", dataset_id, "train.jsonl")
    files = {"train": dataPath}
    ds = load_dataset("json", data_files=files)
    ds = ds.select_columns(["text", "text_pos"])

    save_ds_as_json(ds, "pairs", dataset_id)

if __name__ == "__main__":
    task_mapper=[
        "xzfaq-quality",
        "gcls",
        "senti",
        "scene",
        "hy-bx",
        "hy-dn",
        "hy-hfp",
        "hy-kt",
        "hy-sjpj",
        "hy-yzrsq",
        "hy-xyj",
        "hy-cfxd",
        "hy-dspb",
        "hy-nfyyp",

        "clue/iflytek",
        "clue/tnews",
    ]
    

    for task in task_mapper:
        transform_one(task)