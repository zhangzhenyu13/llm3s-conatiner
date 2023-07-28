import random
import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
import tqdm, json
from processors.data_utils import get_raw_dataset


english_list = [
    "Dahoas/rm-static", "Dahoas/full-hh-rlhf", "Dahoas/synthetic-instruct-gptj-pairwise",
    "yitingxie/rlhf-reward-datasets",
    "openai/webgpt_comparisons",
    'stanfordnlp/SHP',
    "tasksource/oasst1_pairwise_rlhf_reward",
    "andersonbcdefg/dolly_reward_modeling_pairwise",
    "andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai",
    "Anthropic/hh-rlhf",
]

zh_list = [
    "wangrui6/Zhihu-KOL",
    "Cohere/miracl-zh-queries-22-12",
]
dataset_list = english_list

def data_to_json():
    key= "dev"
    for dataset_id in dataset_list:
        print('dataset:', dataset_id )
        print("=======")
        
        loadPath = os.path.join(os.environ['HOME'], 'SharedData/RLHF', "dataset", dataset_id)
        savePath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/{key}', 
                                dataset_id.replace("/","-"), )

        # ds:Dataset = load_from_disk(loadPath)
        # ds.to_json(savePath, num_proc=32,force_ascii=False, batch_size=1000)
        rawdata = get_raw_dataset(dataset_id, "", "", "")
        if "train" == key:
            ds = rawdata.get_train_data()
        else:
            ds = rawdata.get_eval_data()
        if ds is None:
            print("not availabel:", [ key, dataset_id])
            continue
        
        records = []
        for sample in tqdm.tqdm(ds):
            reject = rawdata.get_rejected(sample)
            if reject is None:
                continue
            reject = reject.strip()
            prompt = rawdata.get_prompt(sample).strip()
            if prompt.startswith("Human:"):
                prompt = prompt.lstrip("Human:")
            if prompt.endswith("Assistant:"):
                prompt = prompt.rstrip("Assistant:")
            prompt = prompt.strip()
            chosen = rawdata.get_chosen(sample).strip()
            records.append({
                "prompt": prompt,
                "chosen": chosen,
                "reject": reject
            })
        if len(records) ==0:
            print("skip:", dataset_id)
            continue
        with open(savePath, "w") as f:
            f.writelines(map(lambda x: json.dumps(x, ensure_ascii=False)+"\n", records ) ) 
            

def sample_json():
    key= "train"
    nSamples = 10000
    texts = set()
    com_tag = " |COMBINE__TAG| "
    def extract_texts(lines, add_prompt=True):
        for line in tqdm.tqdm(lines):
            x = json.loads(line)
            prompt = x['prompt']
            chosen = x['chosen']
            reject = x['reject']
            if add_prompt== False:
                # print("add 1")
                line_ext = com_tag.join([chosen, reject])
            else:
                # print("add 2")
                line_ext = com_tag.join([prompt ,chosen, reject])
            texts.add(line_ext)
        

    for dataset_id in dataset_list:
        print('dataset:', dataset_id )
        print("=======")
        
        loadPath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/{key}', 
                                dataset_id.replace("/","-"), )
        
        savePath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/samples', 
                                dataset_id.replace("/","-"), )
        
        lines = []
        with open(loadPath) as f:
            for line in tqdm.tqdm(f):
                lines.append(line)
        lines = random.sample(lines, min(nSamples, len(lines)))

        extract_texts(lines, "Anthropic/hh-rlhf" not in dataset_id)
        with open(savePath, 'w') as fw:
            fw.writelines(lines)
    
        print("extracted:", len(texts))


    savePath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/samples-prompt.json', )
    prompt_translate = []
    def parse_prompt(txt):
        return f"translate following content to Chinse:\n\n{txt}"
    for line_com in texts:
        val_list = line_com.split(com_tag)
        assert len(val_list) in (2, 3)        
        for val in val_list:
            if val.strip():
                prompt_translate.append(parse_prompt(val))
    
    print("list:",len(prompt_translate))
    prompt_translate= set(prompt_translate)
    print("set:", len(prompt_translate))
    with open(savePath, 'w') as fw:
        for idx, x in enumerate(prompt_translate):
            fw.write(json.dumps({"qid": f"qid={idx}", "prompt": x}, ensure_ascii=False)+"\n")


def translate_sample_json():
    en_zh_file = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/', 
                                "samples-prompt-preds.json" )
    en_zh_mapper = {}
    with open(en_zh_file) as f:
        prefix= "translate following content to Chinse:\n\n"
        for line in f:
            x= json.loads(line)
            en_txt= x['prompt'].lstrip(prefix)
            zh_txt = x['pred']
            en_zh_mapper[en_txt] = zh_txt
        en_zh_mapper[""] = ""
        en_zh_mapper["Here is a dialogue:"]= "这是一个对话"
    

    for dataset_id in dataset_list:
        print('dataset:', dataset_id )
        print("=======")
        
        
        loadPath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/samples', 
                                dataset_id.replace("/","-"), )
        savePath = os.path.join(os.environ['HOME'], f'SharedData/Bloom7Bz/reward/samples-translated', 
                                dataset_id.replace("/","-"), )
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        
        with open(loadPath) as f, open(savePath, 'w') as fw:
            for line in tqdm.tqdm(f, desc=f"{dataset_id}"):
                x= json.loads(line)
                prompt , chosen, reject = x['prompt'], x['chosen'], x['reject']
                
                if (prompt not in en_zh_mapper) \
                    or reject not in en_zh_mapper or chosen not in en_zh_mapper:
                    continue
                
                prompt = en_zh_mapper[prompt]
                chosen = en_zh_mapper[chosen]
                reject = en_zh_mapper[reject]
                x= {
                    "prompt": prompt, "chosen": chosen, "reject": reject
                }
                fw.write(json.dumps(x, ensure_ascii=False)+"\n")

        print()


    
            
def pangu_json():
    def process_baike(x):
        
        return x
    def process_couplets(x):
        prompt = f"请你对个对联。\n{x['prompt']}"
        x= {
            'prompt': prompt, "chosen": x['chosen'], 'reject': x['reject']
        }
        return x
    def process_classical(x):
        if "现代文：" in x['prompt']:
            prompt = f"请你将现代文翻译为古文。\n{x['prompt']}"
        elif "古文：" in x['prompt']:
            prompt = f"请你将古文翻译为现代文。\n{x['prompt']}"
        else:
            raise ValueError(f"{x['prompt']}")
        x= {
            'prompt': prompt, "chosen": x['chosen'], 'reject': x['reject']
        }
        return x
    def process_weibo_summary(x):
        prompt = f"请你对新闻内容进行合理地评论。\n{x['prompt']}"
        x= {
            'prompt': prompt, "chosen": x['chosen'], 'reject': x['reject']
        }
        return x
    def process_zhidao(x):
        prompt , chosen , reject = x['prompt'], x['chosen'], x['reject']
        assert "问题：" in prompt and "回答：" in chosen and "回答：" in reject, f"{x}"
        prompt = prompt.replace("问题：","")
        chosen = chosen.replace("回答：", "")
        reject = reject.replace("回答：", "")
        x= {
            'prompt': prompt.replace("问题：",""), "chosen": chosen, 'reject': reject
        }
        return x
    processors= {
        "baike": process_baike,
        "weibo": process_weibo_summary,
        "zhidao": process_zhidao,
        "classical": process_classical,
        "couplets": process_couplets,
    }
    folderPath = os.path.join(os.environ['HOME'], "SharedData/reward-pangu")
    savePath = os.path.join(os.environ['HOME'], "SharedData/Bloom7Bz/reward/pangu")
    if os.path.exists(savePath)== False:
        os.makedirs(savePath)
    dataset_list = os.listdir(folderPath)
    def extract_samples(example, task):
        answers:list = example['answers']
        prompt = example['prompt']
        prefix_response = example['prefix']
        results = []
        answers.sort(key=lambda x: float(x['score']), reverse=True)
        
        for i in range(len(answers)):
            chosen = answers[i]['answer']
            score_1 = float(answers[i]['score'])
            
            for j in range(i+1, len(answers)):
                reject = answers[j]['answer']
                score_2 = float(answers[j]['score'])

                if score_1> score_2:
                    x= {
                            "prompt": f"{prompt}",
                            "chosen": f"{prefix_response}{chosen}",
                            "reject": f"{prefix_response}{reject}"
                        }
                    if task in processors:
                        x = processors[task](x)
                    results.append(
                        x
                    )
            
            if len(results) == 0:
                x= {
                        "prompt": f"{prompt}",
                        "chosen": f"{chosen}",
                        "reject": ""
                    }
                results.append(x)

        return results


    def get_taskname(dataset_id):
        for key in processors:
            if key in dataset_id:
                print("found task:", key)
                return key
        print("task not found")
        return None
    for dataset_id in dataset_list:
        # if "train" not in dataset_id and "dev" not in dataset_id:
        #     continue
        if "train" in dataset_id or "dev" in dataset_id:
            print("skip dataset:", dataset_id)
            continue
        taskname = get_taskname(dataset_id)
        if taskname is None:
            print("skip task/data:", dataset_id)
            continue
        with open(os.path.join(folderPath, dataset_id), "r") as f, \
            open(os.path.join(savePath, dataset_id), 'w') as fw:
            for line in tqdm.tqdm(f, desc=f"{dataset_id}"):
                example = json.loads(line)
                for x in extract_samples(example, taskname):
                    fw.write(json.dumps(x, ensure_ascii=False)+"\n")
        

def load_json_and_save_ds():
    import sys
    infile = sys.argv[1]
    saveKey = sys.argv[2]
    outpath = os.path.join(os.path.dirname(infile), saveKey)
    ds = load_dataset("json", data_files={
        "train": os.path.join(infile, 'train.json'),
        "dev": os.path.join(infile, 'dev.json')})
    ds.save_to_disk(outpath)

# load_json_and_save_ds()

# sample_json()
# data_to_json()

pangu_json()
# translate_sample_json()