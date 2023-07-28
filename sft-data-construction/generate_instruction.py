
import time
import json
import os
import random
import re

import numpy as np
import tqdm
import requests

import io_utils

import fire
from gensim.summarization import bm25
from  transformers import AutoTokenizer
checkpoint = "bigscience/bloomz-7b1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class OpenAIRequestPortal:
    host:str= None
    port:str=None
    method:str=None
    headers={
        'Content-Type': 'application/json'
    }
    @staticmethod
    def request(prompt_list, api_name, model_name, request_batch_size, max_tokens):
        host = OpenAIRequestPortal.host
        port = OpenAIRequestPortal.port
        method = OpenAIRequestPortal.method

        request_batch_size = min(request_batch_size, len(prompt_list))
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
        

def encode_prompt(prompt_instructions, prompt_prior=""):
    """Encode multiple prompt instructions into a single string."""
    prompt = prompt_prior

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<无输入>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. 指令: {instruction}\n"
        prompt += f"{idx + 1}. 输入:\n{input}\n"
        prompt += f"{idx + 1}. 输出:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. 指令:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    try: #for gpt-3.5-turbo
        raw_instructions = response["message"]["content"]
    except:
        try:
            raw_instructions = response["text"]  #for text-davinci-003
        except:
            print("ERROR parse!")
    if '指令:' not in raw_instructions[0: 10] and '指令：' not in raw_instructions[0: 10]:
        raw_instructions = f"{num_prompt_instructions+1}. 指令:" + raw_instructions
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    blacklist = ["图像", "图片", "照片", "文件", "图表", "图层", "曲线图", "折线图", "直线图", "柱形图", "饼状图", "链接", "http",'OpenAI', 'chatgpt', 'gpt-3', 'gpt-3.5', 'gpt-4']
    replace_empty_list = ['要求GPT模型能够', '要求GPT能够', '要求GPT模型', '让GPT模型', '使用GPT模型', '请向GPT模型', 'GPT模型应', 'GPT模型应该', '请求GPT模型', '需要GPT模型回答', '请GPT模型'
                          , '请让GPT模型', '训练GPT模型', 'GPT模型需要', '要求GPT', '让GPT', '使用GPT', '请向GPT', 'GPT应', 'GPT应该', '请求GPT', '需要GPT回答', '请GPT', '请让GPT'
                          , '训练GPT', 'GPT需要', '希望GPT模型能够', '希望GPT能够', '以便GPT模型能够', '以便GPT能够', '使得GPT模型能够', '使得GPT能够', '使GPT模型能够', '使GPT能够'
                          , '由GPT模型', '使GPT模型']
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        intruction_pattern = re.compile(r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")")
        input_pattern = re.compile(r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")")
        output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
        intruction_match = intruction_pattern.search(inst)
        input_match = input_pattern.search(inst)
        output_match = output_pattern.search(inst)
        if intruction_match and input_match and output_match:
            inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n')
            input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n')
            input = "" if "无输入" in input else input
            output = output_match.group().strip().strip('\n')
            if '指令:' in output and '输入:' in output and '输出:' in output: # 返回若没有以###号区分，取第一条数据
                output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
                output_match_new = output_pattern_new.search(output)
                if output_match_new:
                    output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n')
            # 去掉不合理的instruction
            if len(inst) <= 3:
                continue
                
            for item in replace_empty_list:
                inst = inst.replace(item, "") 
            
            if "GPT" in inst or 'GPT' in input:
                continue
                
            if len(input) == 0:  # input无输入
                instructions.append({"instruction": inst, "input": input, "output": output})
            else:
                if '示例' in inst or '例子' in inst:  # inst里给例子
                    if len(inst) < 150:
                        instructions.append({"instruction": inst, "input": input, "output": output})
                else:  # 没给例子
                    if len(inst) < 100:
                        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return w in s


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./zh_seed_tasks.json",
    num_instructions_to_generate=1,
    api="completion",
    model_name="text-davinci-003",
    prompt_path="./xxxx",
    service_host="",
    service_port="",
    service_method="",
    num_prompt_instructions=3,
    num_tokens=128,
    request_batch_size=1,
    
):  
    OpenAIRequestPortal.host= service_host
    OpenAIRequestPortal.port= service_port
    OpenAIRequestPortal.method = service_method

    prompt_prior = []
    with open(prompt_path, 'r') as f:
        for line in f:
            if line.strip()!="":
                prompt_prior.append(line)
    prompt_prior = "".join(prompt_prior)
    print(f"using following prompt prior:\n{prompt_prior}")

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")


    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen_iter.json")):
        machine_instruction_data = io_utils.jload(os.path.join(output_dir, "data.train.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")


    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [tokenizer.tokenize(inst) for inst in all_instructions]
    bm25Model = bm25.BM25(all_instruction_tokens)


    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)#    seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, prompt_prior=prompt_prior)
            batch_inputs.append(prompt)
        
        request_start = time.time()
        
        results = OpenAIRequestPortal.request(
            prompt_list=batch_inputs,
            api_name=api,
            model_name=model_name,
            request_batch_size=request_batch_size,
            max_tokens=int(num_tokens)
            )
        results = results['generations']
        
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = tokenizer.tokenize(instruction_data_entry["instruction"])
            rouge_scores = bm25Model.get_scores(new_instruction_tokens)

            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) >18:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        io_utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
