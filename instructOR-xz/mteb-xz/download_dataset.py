import sys
import csv
csv.field_size_limit(sys.maxsize)

import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset

os.environ['http_proxy'] = "socks5h://127.0.0.1:1080"
os.environ['https_proxy'] = "socks5h://127.0.0.1:1080"
dataset_list = [
    # ('THUIR/T2Ranking', 'collection'),
    # ('THUIR/T2Ranking', 'queries.dev'),
    # ('THUIR/T2Ranking', 'qrels.dev'),
    'vegaviazhang/Med_QQpairs',
    ('clue', 'tnews'),
    ('tyqiangz/multilingual-sentiments', 'chinese'),
    ('clue', 'iflytek'),
    ('kuroneko5943/jd21', 'iPhone'),
    ('kuroneko5943/stock11', 'communication'),
    'Fearao/guba_eastmoney',
]
def data_cache_local():
    for dataset_id in dataset_list:
        print('dataset:', dataset_id )
        if isinstance(dataset_id, tuple):
            dataset = load_dataset(*dataset_id )
            dataset_id = "/".join(dataset_id)
        else:
            dataset = load_dataset(dataset_id )
        print(dataset)
        print("=======")

        if isinstance(dataset, DatasetDict):
            data = DatasetDict()
            for key in dataset.keys():

                savePath = os.path.join(os.environ['HOME'], 'SharedData/mteb_zh', dataset_id, key)
                dataset[key].save_to_disk(savePath)
                print(key, savePath)
                ds =  load_from_disk(savePath)
                print(ds)
                data[key] = ds
            print(data)
            print("********")
        else:
            savePath = os.path.join(os.environ['HOME'], 'SharedData/mteb_zh', dataset_id, "data")
            dataset.save_to_disk(savePath)

data_cache_local()