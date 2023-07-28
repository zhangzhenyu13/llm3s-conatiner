import json
import requests
import regex
import random

class OpenAIRequestPortal:
    host:str= "server-ip"
    port:str="1239"
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



def simple_test():
    prompts = [
        "test 1: can you spell a word longer than 10"
        "test2: describe Baiden"
    ]

    result = OpenAIRequestPortal.request(
        prompts, "chat","gpt-3.5-turbo", #"text-davinci-003",
        2, 128
    )
    print(result)

def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_cn.txt").read() + "\n"
    # print(prompt_instructions, type(prompt_instructions[0]))

    for idx, task_dict in enumerate(prompt_instructions):
        # print(task_dict)
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = regex.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<无输入>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. 指令: {instruction}\n"
        prompt += f"{idx + 1}. 输入:\n{input}\n"
        prompt += f"{idx + 1}. 输出:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. 指令:"
    return prompt
def instructions_generate():
    
    seed_tasks = [json.loads(l) for l in open('./zh_seed_tasks.json', "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    # print(seed_instruction_data[0], type(seed_instruction_data[0]))
    request_prompts = []
    sampled_instructions = []
    request_batch_size = 2
    instructions_size_per_sample = 3
    for _ in range(request_batch_size):
        sampled_instructions.append(
            random.sample(seed_instruction_data, instructions_size_per_sample )
        )
        prompt_sample = encode_prompt(sampled_instructions[-1]) 
        request_prompts.append(prompt_sample)
    

    # return
    results = OpenAIRequestPortal.request(
        prompt_list= request_prompts, 
        api_name='completion',
        model_name='text-davinci-003', 
        request_batch_size=request_batch_size,
        max_tokens=1024
    )

    print(results)

    for i in range(2):
        print('Examples:#', i)
        print(seed_instruction_data[i])
        print(request_prompts[i])
        print(results['generations'][i])
        print()
        print()







if __name__ == "__main__":
    simple_test()
    # instructions_generate()
