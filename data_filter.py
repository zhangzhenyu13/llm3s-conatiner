import datasets
from datasets import load_dataset
import os
from transformers import AutoTokenizer
cutoff_len = 512
tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.environ['HOME'], "CommonModels/bigscience/bloomz-560m"))

def process():
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            return_attention_mask=True
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= cutoff_len:
            result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][cutoff_len - 1] = 1

        return result

    def generate_and_tokenize_prompt(data_point):
        prompt = data_point['prompt']
        chosen = data_point['chosen']
        reject = data_point['reject']

        prompt_chosen = "User: " + prompt + "\n\nBot: " + chosen + tokenizer.eos_token
        prompt_reject = "User: " + prompt + "\n\nBot: " + reject + tokenizer.eos_token
        if tokenizer.bos_token_id != None:
            prompt_chosen = tokenizer.bos_token + prompt_chosen
            prompt_reject = tokenizer.bos_token + prompt_reject

        tokenized_chosen_prompt = tokenize(prompt_chosen)
        tokenized_reject_prompt = tokenize(prompt_reject)
        # print("tuning-->",tokenized_chosen_prompt.keys(), tokenized_reject_prompt.keys())
        
        return {
            "chosen_input_ids": tokenized_chosen_prompt['input_ids'],
            "chosen_attention_mask": tokenized_chosen_prompt['attention_mask'],
            "reject_input_ids": tokenized_reject_prompt['input_ids'],
            "reject_attention_mask": tokenized_reject_prompt['attention_mask'],
            "chosen-len": len(tokenized_chosen_prompt['input_ids']),
            "reject-len": len(tokenized_reject_prompt['input_ids'])
        }
    datafiles = {"train": os.path.join(os.environ['HOME'], "SharedData/Bloom7Bz/reward/train-d4.json")}
    rawdataset= load_dataset("json", data_files=datafiles)
    rawdataset = rawdataset.map(generate_and_tokenize_prompt, 
                                desc="processing-len",
                                num_proc=40
                    )
    print(rawdataset)
    rawdataset = rawdataset.filter (lambda x: max(x['chosen-len'], x['reject-len'] ) < 520)

    print(rawdataset)
    rawdataset  = rawdataset.remove_columns(["chosen-len", "reject-len"])

    rawdataset['train'].to_json(os.path.join(os.environ['HOME'], "SharedData/Bloom7Bz/reward/train-d4-filtered.json"),
                    #    num_proc=32,
                    force_ascii=False
                       )

process()