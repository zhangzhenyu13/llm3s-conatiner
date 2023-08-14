import os
localPath = os.path.join(os.environ['HOME'], "SharedData/instructOR")
from datasets import load_dataset, DatasetDict, load_from_disk, concatenate_datasets
from dataloader import load_dataset_local

os.environ['http_proxy'] = 'socks5h://127.0.0.1:1080'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:1080'

datalist =[
    # 'vegaviazhang/Med_QQpairs',
    # 'cmrc2018',
    # 'wangrui6/Zhihu-KOL',
    # ('Hello-SimpleAI/HC3-Chinese', 'all'),
    # ('wiki_atomic_edits', 'chinese_insertions'),
    # ('wiki_atomic_edits', 'chinese_deletions'),
    # 'michaelwzhu/ChatMed_Consult_Dataset',
    # 'michaelwzhu/ShenNong_TCM_Dataset',
    # ('amazon_reviews_multi', 'zh'),
    # ('csebuetnlp/xlsum', 'chinese_simplified'),
    # ('mlqa', 'mlqa-translate-train.zh'),
    # ("clue","afqmc"),
    # ("clue","c3"),
    # ("clue","chid"),
    # ("clue","cluewsc2020"),
    # ("clue","cmnli"),
    # ("clue","csl"),
    # ("clue","drcd"),
    # ("clue","iflytek"),
    # ("clue","ocnli"),
    # ("clue","tnews"),
    # ('shibing624/nli_zh', 'BQ'),
    # ('shibing624/nli_zh', 'LCQMC'),
    # ('shibing624/nli_zh', 'PAWSX'),
    # ('shibing624/nli_zh', 'ATEC'),
    # ('shibing624/nli_zh', 'STS-B'),
    # ('suolyer/webqa'),
    # ('neuclir/csl'),
    # ('PaddlePaddle/dureader_robust'),
    # ('miracl/miracl-corpus', 'zh'),
    # ('YeungNLP/firefly-train-1.1M'),
    # ('shibing624/alpaca-zh'),
    # "shibing624/snli-zh",
    "shibing624/sts-sohu2021"


    # 'openchat/openchat_sharegpt4_dataset',
    # 'BelleGroup/train_3.5M_CN',
    # 'BelleGroup/generated_chat_0.4M',
    # 'BelleGroup/school_math_0.25M',
    # 'BelleGroup/train_2M_CN',
    # 'BelleGroup/train_1M_CN',
    # 'BelleGroup/train_0.5M_CN',
    # 'BelleGroup/multiturn_chat_0.8M',
    # 'THUDM/webglm-qa'
   
    # 'WizardLM/WizardLM_evol_instruct_V2_196k',
    # 'stingning/ultrachat',
    # 'fnlp/moss-003-sft-data',


]

def checkdata_info(dataset_id):
    print('dataset:', dataset_id )
    try:
        dataset = load_dataset_local(dataset_id, False )
    except Exception as e:
        print(e.args)
        print()
        return
    print(dataset)
    print("=======")
    print()


def download(dataset_id):
    print('dataset:', dataset_id )
    if isinstance(dataset_id, str):
        if dataset_id == "shibing624/sts-sohu2021":
            dslist =["dda","ddb", "dca", "dcb", "cca", "ccb"]
            dslist=[load_dataset(dataset_id, dsid) for dsid in dslist]
            dataset = DatasetDict(
                train= concatenate_datasets([ds['train'] for ds in dslist]),
                test= concatenate_datasets([ds['test'] for ds in dslist])
            )
        else:
            dataset = load_dataset(dataset_id )
    else:
        dataset = load_dataset(*dataset_id )
        dataset_id = "/".join(dataset_id)
    print(dataset)
    print("=======")

    if isinstance(dataset, DatasetDict):
        data = DatasetDict()
        for key in dataset.keys():

            savePath = os.path.join(localPath, dataset_id, key)
            dataset[key].save_to_disk(savePath)
            print(key, savePath)
            ds =  load_from_disk(savePath)
            print(ds)
            data[key] = ds
        print(data)
        print("********")
    else:
        savePath = os.path.join(localPath, dataset_id, "train")
        dataset.save_to_disk(savePath)
        ds =  load_from_disk(savePath)
        print(ds)


def downloadXZ(dataset_id):
    print('dataset:', dataset_id )
    folderPath = os.path.join(os.environ['HOME'], "SharedData/pretrain/tasks")
    files = {
        "train": os.path.join(folderPath, dataset_id, "train.json"),
        "dev": os.path.join(folderPath, dataset_id, "dev.json"),
        "test": os.path.join(folderPath, dataset_id, "test.json")
    }
    rm_fn = []
    for fn in files:
        if not os.path.exists(files[fn]):
            print([fn, dataset_id])
            rm_fn.append(fn)
    [files.pop(fn) for fn in rm_fn]

    dataset = load_dataset("json", data_files=files)
    print(dataset)
    print("=======")

    if isinstance(dataset, DatasetDict):
        data = DatasetDict()
        for key in dataset.keys():

            savePath = os.path.join(localPath, dataset_id, key)
            dataset[key].save_to_disk(savePath)
            print(key, savePath)
            ds =  load_from_disk(savePath)
            print(ds)
            data[key] = ds
        print(data)
        print("********")
    else:
        savePath = os.path.join(localPath, dataset_id, "train")
        dataset.save_to_disk(savePath)
        ds =  load_from_disk(savePath)
        print(ds)


for dataname in datalist:
    download(dataname)
    checkdata_info(dataname)

# for dataname in xzdata:
#     # downloadXZ(dataname)
#     checkdata_info(dataname)
