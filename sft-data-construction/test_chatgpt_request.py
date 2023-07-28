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

def load_json(infile, prompt_key="text"):
    return [json.loads(l)[prompt_key] for l in open(infile, "r")]

def result_generate(infile, outfile):
    prompt_list = load_json(infile)
    records = []

    for i in tqdm.trange(len(prompt_list)):
        print(prompt_list[i])
        results = OpenAIRequestPortal.request(
            prompt_list= prompt_list[i], 
            api_name='chat',
            model_name='gpt-3.5-turbo', 
            request_batch_size=1,
            max_tokens=2048
        )

        print(results)
        
        preds = results['generations']['message']['content']
        x={"pred_text": preds}
        records.append(x)
    
    with open(outfile, 'w') as fw:
        fw.writelines(map(lambda x: json.dumps(x, ensure_ascii=False)+"\n", records))
    
        
        

if __name__ == "__main__":
    import sys
    result_generate(*sys.argv[1:])
