import random
from datasets import DatasetDict, Dataset, concatenate_datasets
from instructions_emb import instructionsMapper
import os
import sys
import regex
sys.path.append(os.path.dirname(
    os.path.dirname(
        os.path.abspath(__name__)
    )
))
from dataloader import load_dataset_local


outPath ="/export2/scratch/zhangzhenyu/SharedData/instructOR-datasets"

def ds_map_func(ds:DatasetDict, map_func)->DatasetDict:
    return ds.map(
        map_func, 
        num_proc=max(10,min(len(ds)//30000, 32)),
    )

def concate_ds_dict(dslist):
    dataset_dict = {
        key: concatenate_datasets([ds[key] for ds in dslist])
        for key in dslist[0].keys()         
    }
    return dataset_dict

def filter_positives(dataset_dict: DatasetDict):
    ds = dataset_dict['train']
    if "label" in ds.column_names:
        return dataset_dict.filter(lambda x: x['label']== 1 
                    ).select_columns(["text", "text_pos"])
    return dataset_dict

def save_ds_as_json(dataset_dict: DatasetDict, data_format, dataset_id ):
    assert data_format in ("triplets", "pairs", "tripletsI", "pairsI")

    match data_format:
        case "triplets"| "tripletsI": 
            dataset_dict = dataset_dict.select_columns(["text", "text_pos", "text_neg"])
        case "pairs" | "pairsI": 
            dataset_dict= dataset_dict.select_columns(["text", "text_pos"])
        case _:
            raise ValueError(f"unknown ds format:{data_format}")

    if not isinstance(dataset_id, str):
        dataset_id= "/".join(dataset_id)


    for key in dataset_dict:
        print("saving...", key)
        ds:Dataset = dataset_dict[key]
        ds.to_json(
            os.path.join(outPath, data_format, dataset_id, key+".jsonl"), 
            force_ascii=False,
            num_proc=max(1,min(len(ds)//100000, 32))
        )

def load_clue_csl(dataset_id='clue/csl'):
    def func_mapper(x):
        keywords = x['keyword']
        text = x['abst']
        keywords = ", ".join(keywords)
        return {
            "text": keywords, "text_pos": text
        }
    
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict.pop("test")
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    dataset_dict = ds_map_func(dataset_dict, func_mapper)
    print(dataset_dict)
    print(dataset_dict.column_names)
    dataset_dict = dataset_dict.select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_csl_abt_title(dataset_id='neuclir/csl'):
    def kw_mapper(x):
        keywords = x['keywords']
        text = x['abstract']
        keywords = ", ".join(keywords)
        return {
            "text": keywords, "text_pos": text
        }
    def title_mapper(x):
        title = x['title']
        text = x['abstract']
        return {
            "text": title, "text_pos": text
        }
    
    dataset_dict = load_dataset_local(dataset_id)
    
    dataset_dict1 = ds_map_func(dataset_dict, kw_mapper
            ).select_columns(["text", "text_pos"])
    dataset_dict2 =  ds_map_func(dataset_dict, title_mapper
            ).select_columns(["text", "text_pos"])

    dataset_dict = concate_ds_dict([dataset_dict1, dataset_dict2])
    
    dataset_dict= DatasetDict(train= dataset_dict['csl'])

    
    return dataset_dict

def load_c3(dataset_id='clue/c3'):
    def func_mapper(x):
        context = x['context']
        question = x['question']
        context = "\n".join(context)
        return {
            "text": question, "text_pos": context
        }
    
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = ds_map_func(dataset_dict, func_mapper
        ).select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_webqa(dataset_id='suolyer/webqa'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'input': 'text', 'output': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    
    return dataset_dict


def load_cmrc2018(dataset_id='cmrc2018'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_medqq(dataset_id='vegaviazhang/Med_QQpairs'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'question1': 'text', 'question2': 'text_pos'}
        ).select_columns(["text", "text_pos", "label"])
    return dataset_dict

def load_nli(dataset_id="shibing624/snli-zh"):
    def change_label(x):
        r = x['r']
        if r in ('entailment', 'neutral', 'contradiction'):
            if r== 'entailment':
                x['label'] =1
            else:
                x['label'] =0
        else:
            x['label'] = int(r ==0 )
        return x
    
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'premise': 'text', 'hypothesis': 'text_pos', 'label': "r"}
        )
    dataset_dict = ds_map_func(dataset_dict, change_label
        ).select_columns(["text", "text_pos", "label"])
    return dataset_dict

def load_qq(dataset_id='shibing624/nli_zh/PAWSX'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'}
        ).select_columns(["text", "text_pos", "label"])
    return dataset_dict

def load_xzqq(dataset_id='xzqq'):
    def label_change(x):
        label = x['r']
        if label in ("0", "1"):
            label = int(label)
        elif label in ("good", "bad"):
            label = int (label =="good")
        else:
            raise ValueError(f"label not supported:{x}")
        x['label'] = label
        return x
    
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_column("label", "r")
    dataset_dict = ds_map_func(dataset_dict, label_change)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'}
        ).select_columns(["text", "text_pos", "label"])
    return dataset_dict


def load_wiki_edit(dataset_id='wiki_atomic_edits/chinese_insertions'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'base_sentence': 'text', 'edited_sentence': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    return dataset_dict

def load_h3c_qa(dataset_id='Hello-SimpleAI/HC3-Chinese'):
    def transform_ans_format(x):
        return {
            "h-a": x['human_answers'][0],
            "c-a": x['chatgpt_answers'][0]
        }
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = ds_map_func(dataset_dict, transform_ans_format)
    dataset_dict1 = dataset_dict.rename_columns({'question': 'text', 'h-a': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    dataset_dict2 = dataset_dict.rename_columns({'question': 'text', 'c-a': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    
    dataset_dict = {
        key: concatenate_datasets([dataset_dict1[key], dataset_dict2[key],])
        for key in dataset_dict.keys()         
    }
    dataset_dict= DatasetDict(**dataset_dict).filter(
        lambda x: x['text_pos'] is not None
        )
    return dataset_dict

def load_qr_no_label(dataset_id='michaelwzhu/ChatMed_Consult_Dataset'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'query': 'text', 'response': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    return dataset_dict

def load_review_multi(dataset_id='amazon_reviews_multi/zh'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'review_title': 'text', 'review_body': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    return dataset_dict


def load_title_text(dataset_id='miracl/miracl-corpus/zh'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'title': 'text', 'text': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    return dataset_dict


def load_article_summary(dataset_id='csebuetnlp/xlsum/chinese_simplified'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_column("text", "passage")
    dataset_dict1 = dataset_dict.rename_columns({'title': 'text', 'summary': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    dataset_dict2 = dataset_dict.rename_columns({'title': 'text', 'passage': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    dataset_dict = {
        key: concatenate_datasets([dataset_dict1[key], dataset_dict2[key],])
        for key in dataset_dict.keys()         
    }

    return dataset_dict

def load_zhikuKL_inst(dataset_id='wangrui6/Zhihu-KOL'):
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_columns({'INSTRUCTION': 'text', 'RESPONSE': 'text_pos'}
        ).select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_belle_inst(dataset_id='BelleGroup/train_2M_CN'):
    def concate_instruction(x):
        text = x['instruction'] + (
            f"\n{x['input']}" if x['input'].strip() else "")
        text_pos = x['output']
        return {
            "text": text, "text_pos": text_pos
        }
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = ds_map_func(dataset_dict, concate_instruction
            ).select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_firely_inst(dataset_id='YeungNLP/firefly-train-1.1M'):
    def concate_instruction(x):
        text = x['input']
        text_pos = x['target']
        return {
            "text": text, "text_pos": text_pos
        }
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = ds_map_func(dataset_dict, concate_instruction
            ).select_columns(["text", "text_pos"])
    
    return dataset_dict

def load_ultrachat(dataset_id='stingning/ultrachat'):
    dataset_dict = load_dataset_local(dataset_id)
    def conv_split(datapoint):
        sess = datapoint['data']
        samples = []
        context= ""

        for i, u in enumerate(sess):
            role = "assistant" if i%2==1 else "human"
            
            if role == "assistant" and context:
                samples.append({
                    "text": context,
                    "text_pos": role +": " + u
                })
            if context:
                context += "[EOT]\n" + role +": " + u
            else:
                context = role +": " + u
        return random.sample(samples, k=1)[0]


    dataset_dict = ds_map_func( dataset_dict, conv_split
            ).select_columns(["text", "text_pos"])
    
    print(dataset_dict)

    return dataset_dict

def load_belle_conv(dataset_id='BelleGroup/train_3.5M_CN'):
    dataset_dict = load_dataset_local(dataset_id)
    def conv_split(datapoint):
        sess = datapoint['conversations']
        samples = []
        context= ""

        for i, x in enumerate(sess):
            role = x['from']
            u = x['value']
            
            if role == "assistant" and context:
                samples.append({
                    "text": context,
                    "text_pos": role +": " + u
                })

            if context:
                context += "[EOT]\n" + role +": " + u
            else:
                context = role +": " + u
        return random.sample(samples, k=1)[0]


    dataset_dict = ds_map_func( dataset_dict, conv_split
            ).select_columns(["text", "text_pos"])
    

    return dataset_dict


def load_moss003_conv(dataset_id='fnlp/moss-003-sft-data'):
    dataset_dict = load_dataset_local(dataset_id)
    reg_role = regex.compile(f"<|MOSS|>:|<|Human|>:")
    def conv_split(datapoint):
        sess = sorted(datapoint['chat'].items())
        assert sess[0][0] == 'turn_1'
        sess = filter(lambda x: x[1] is not None, sess)
        sess = list(map(lambda x: x[1], sess ) )

        samples = []
        context= ""

        for i, x in enumerate(sess):
            uhuman= x['Human']
            uassistant = x['MOSS']
            uhuman = reg_role.sub("", uhuman).strip()
            uassistant = reg_role.sub("", uassistant).strip()
            context += "human: " + uhuman +"[EOT]\n" 

            if context:            
                samples.append({
                    "text": context,
                    "text_pos": "assistant: " + uassistant
                })
            context += "assistant: " + uassistant +"[EOT]\n" 
            
        return random.sample(samples, k=1)[0]


    dataset_dict = ds_map_func( dataset_dict, conv_split
            ).select_columns(["text", "text_pos"])
    

    return dataset_dict

if __name__ == "__main__":
    useInsturction=False
    dsPart='train'

    task_mapper={
        "cmrc2018": load_cmrc2018,
        "vegaviazhang/Med_QQpairs": load_medqq,
        "shibing624/nli_zh/ATEC": load_qq,
        "shibing624/nli_zh/BQ": load_qq,
        "shibing624/nli_zh/LCQMC": load_qq,
        "shibing624/nli_zh/PAWSX": load_qq,
        "shibing624/nli_zh/STS-B": load_qq,
        "shibing624/sts-sohu2021": load_qq,
        "shibing624/snli-zh": load_nli,
        "wangrui6/Zhihu-KOL": load_zhikuKL_inst,
        "Hello-SimpleAI/HC3-Chinese/all": load_h3c_qa,
        "wiki_atomic_edits/chinese_insertions": load_wiki_edit,
        "wiki_atomic_edits/chinese_deletions": load_wiki_edit,
        "michaelwzhu/ChatMed_Consult_Dataset": load_qr_no_label,
        "michaelwzhu/ShenNong_TCM_Dataset": load_qr_no_label,
        "amazon_reviews_multi/zh": load_review_multi, 
        "csebuetnlp/xlsum/chinese_simplified": load_article_summary,
        "mlqa/mlqa-translate-train.zh": load_cmrc2018,
        "clue/afqmc": load_qq,
        "clue/c3": load_c3,
        "clue/cmnli": load_qq,
        "clue/csl": load_clue_csl,
        "clue/drcd": load_cmrc2018,
        # "clue/iflytek": load_dataset_local,
        "clue/ocnli": load_qq,
        # "clue/tnews": load_dataset_local,
        "suolyer/webqa": load_webqa,
        "neuclir/csl": load_csl_abt_title,
        "PaddlePaddle/dureader_robust": load_cmrc2018,
        "miracl/miracl-corpus/zh": load_title_text,

        "xzqq": load_xzqq,
        "kfqq": load_xzqq,
        "qq": load_xzqq,
        "qr": load_xzqq,


        
        "BelleGroup/train_3.5M_CN": load_belle_conv,
        "BelleGroup/generated_chat_0.4M": load_belle_inst,
        "BelleGroup/school_math_0.25M": load_belle_inst,
        "BelleGroup/train_2M_CN": load_belle_inst,
        "BelleGroup/train_1M_CN": load_belle_inst,
        "BelleGroup/train_0.5M_CN": load_belle_inst,
        "BelleGroup/multiturn_chat_0.8M": load_belle_inst,

        # GPT4-based self-instruct dataset
        "shibing624/alpaca-zh": load_belle_inst,

        # 2-Turbo API based multi-round dialogue mimic(human) data
        "stingning/ultrachat": load_ultrachat,
        # fnlp moss sft dataset
        "fnlp/moss-003-sft-no-tools": load_moss003_conv,
        # Firely dataset
        "YeungNLP/firefly-train-1.1M": load_firely_inst,


    }
    task_mapper= {k: task_mapper[k] 
        for k in ["Hello-SimpleAI/HC3-Chinese/all"]
    }
    for task in task_mapper:
        loader_func = task_mapper[task]
        ds= loader_func(dataset_id= task)
        ds = filter_positives(ds)
        print("after filter:", len(ds[dsPart]))
        print(ds)

        save_ds_as_json(
            ds,
            "pairs",
            task
        )
        
