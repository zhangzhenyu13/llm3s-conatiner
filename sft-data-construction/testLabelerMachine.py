import json
import requests
import regex
import random
import pandas as pd
import tqdm

mapperLabel={"yes": "good", "no":"bad"}
mapperLabelInv={V:K for K, V in mapperLabel.items() }

class OpenAIRequestPortal:
    host:str= "server-ip"
    port:str="1238"
    method:str="ouryx05private"
    headers={
        'Content-Type': 'application/json'
    }
    @staticmethod
    def request(prompt_list, api_name, model_name, request_batch_size, max_tokens):
        host = OpenAIRequestPortal.host
        port = OpenAIRequestPortal.port
        method = OpenAIRequestPortal.method
        
        # prompts = requestData['prompts']
        # api = requestData['api']
        # model_name = requestData['model_name']
        # batch_size = requestData['batch_size']

        payload = json.dumps({
            "prompts": prompt_list, "api": api_name, "model_name": model_name,
            "batch_size": request_batch_size, "max_tokens": max_tokens
        })
        response= requests.post(
            url=f"http://{host}:{port}/{method}",
            data = payload, 
            headers=OpenAIRequestPortal.headers
        )
        results = json.loads(response.text)
        return results

def load_json(infile):
    return [json.loads(l) for l in open(infile, "r")]

def load_tsv(infile):
    cols = ["sentence1", "sentence2", "label"]
    records = []
    with open(infile) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            records.append(dict(zip(cols, fields)))
    return records

def encode_prompt(samples, demos):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("labelersMachine/prompt_qq-v3.txt").read() + "\n"
    def build_instructions(name, data, icl=False):
        if icl:
            prompt_data = "下面是一些示例："+"\n"
        else:
            prompt_data = "下面是一些需要判断的样本："+"\n"
            if len(samples) ==1:
                prompt_data = ""
        labels = []
        for idx, task_dict in enumerate(data):
            # print(task_dict)
            q1, q2 = task_dict["sentence1"], task_dict["sentence2"]
            labels.append(f"{idx+1}:"+mapperLabelInv[task_dict['label']])
            if len(samples)> 1:
                prompt_data += f"{name}{idx + 1}\n"
            prompt_data += f"句子1:{q1}\n"
            prompt_data += f"句子2:{q2}\n"
            if len(samples) > 1:
                prompt_data += f"###\n"

        if len(samples) >1:
            prompt_data += f"判断列表：\n"
        else:
            prompt_data += f"判断结果：\n"
        if icl:
            prompt_data+= "\n".join(labels)+"\n"
        return prompt_data
    
    if len(demos) >0 :
        prompt += build_instructions("示例", demos, True) +"\n"
    prompt += build_instructions("样本", samples, False)
    return prompt
def labelers_generate(demo_file, infile, outfile):
    demo_tasks = load_json(demo_file)
    # seed_tasks = load_json(infile)
    seed_tasks = load_tsv(infile)
    random.seed(2023)
    # seed_tasks = random.sample(seed_tasks, 20)
    demo_df = pd.DataFrame(demo_tasks)
    good_demos = demo_df[demo_df['label']=='good'].to_dict(orient="records")
    bad_demos = demo_df[demo_df['label']=='bad'].to_dict(orient="records")
    print()
    demo_tasks = random.sample(good_demos, 5) + random.sample(bad_demos, 5)
    demo_tasks = []
    seed_tasks = seed_tasks[:20]

    batch_size = 1
    

    # prompt = encode_prompt(seed_tasks, demo_tasks)
    # print(prompt)
    # print(len(prompt))
    # prompt_list = [prompt]

    seed_tasks_list = [seed_tasks[i:i+batch_size] for i in range(0, len(seed_tasks), batch_size) ]
    prompt_list = [encode_prompt(batch_seeds, demo_tasks) for batch_seeds in seed_tasks_list ]
     
    [print(len(prompt)) for prompt in prompt_list]

    request_batch_size = min(len(prompt_list), 4)
    records = []

    for i in tqdm.trange(len(prompt_list)):
        print(prompt_list[i])
        results = OpenAIRequestPortal.request(
            prompt_list= prompt_list[i], 
            api_name='chat',
            model_name='gpt-3.5-turbo', 
            request_batch_size=request_batch_size,
            max_tokens=2048
        )

        print(results)
        
        preds = results['generations']['message']['content'].split("\n")
        print("task nums:", len(preds), len(seed_tasks_list[i]) )
        
        for idx, (pred, x) in enumerate(zip(preds, seed_tasks_list[i])):
            try:
                pred= pred.strip()
                if ":" in pred:
                    idx_out, pred = pred.split(":")
                    assert idx+1 == int(idx_out)
                
                x['pred'] = mapperLabel.get(pred, pred)
                records.append(x)
            except Exception as e:
                print(e.args, pred)
                continue
    
    with pd.ExcelWriter(outfile) as f:
        df = pd.DataFrame(records)
        df.to_excel(f, index=False, sheet_name="test")
        print((df['label']==df['pred']).sum(), len(df), 
                "%.4f"%((df['label']==df['pred']).sum()/ len(df))
        )
        
        

if __name__ == "__main__":
    import sys
    labelers_generate(*sys.argv[1:])
