from fastapi import FastAPI, Request
import uvicorn, json, datetime
from contextlib import asynccontextmanager

import os

import torch
from configs.supported import SupportedModels
from transformers import AutoTokenizer
from reward_model import RewardModel
from peft import PeftModel
if torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("Not Supported")

def get_model(base_model, lora_weights=None):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    print("loading in cuda", base_model)
    model = RewardModel.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        # load_in_8bit = True,
        device_map="auto",
    )
    if lora_weights and os.path.exists(lora_weights):
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    model.eval()
    print('finished loading', base_model)
    return model

class ModelWorker:
    model: RewardModel = None
    tokenizer: AutoTokenizer = None
    max_length = 2048
    roles : list = []

def load_model(model_id="your-org/rewardS1"):
    model_path = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "right" 
   
    ModelWorker.model = get_model(model_path)
    ModelWorker.tokenizer = tokenizer
    if model_id in SupportedModels.roled_model_ids:
        ModelWorker.roles = SupportedModels.roled_model_ids[model_id]
    print(ModelWorker.model)
    print(ModelWorker.tokenizer)
    print("role info:", ModelWorker.roles)


def predict_reward(history, **kwargs ):
    tokenizer, model = ModelWorker.tokenizer, ModelWorker.model

    # ===== begin length process
    prompt = [ ]
    for (old_query, response) in history:
        prompt += [ModelWorker.roles[0] +": "+ old_query, ModelWorker.roles[1] +": "+ response  ]
    
    prompt_length = len(tokenizer.tokenize("\n".join(prompt[:-1]) + f"\n\n{ModelWorker.roles[1]}: " ))
    response_length = len(tokenizer.tokenize(response)) + 1
    prompt = "\n".join(prompt[:-1]) + f"\n\n{ModelWorker.roles[1]}: {response}" + tokenizer.eos_token

    input_length= len(tokenizer.tokenize(prompt))
    if input_length > ModelWorker.max_length:
        print("max context length: ", [input_length, ModelWorker.max_length ])
        return {"score": None, "reason": "Maximum Context Length Reached!"}
            
    prompt = tokenizer.bos_token + prompt if tokenizer.bos_token!=None else prompt
    print("Tokens of prompt, response, input:", prompt_length, response_length, input_length)
    print("EOS:",tokenizer.eos_token)

    print(f"prompt:\n{prompt}\n******")
    # ===== finished length process

    inputs = tokenizer(prompt, return_tensors="pt")
    # print(inputs)

    with torch.no_grad():
        reward_outputs = ModelWorker.model.forward_value(
            input_ids= inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            prompt_length = prompt_length,
            return_value_only=False
        )
    values = reward_outputs['values'].cpu().numpy().tolist()[0]
    reward_score = reward_outputs['chosen_end_scores'].cpu().numpy().tolist()[0]
    reward_score_plus = torch.sigmoid(reward_outputs['chosen_end_scores']).cpu().numpy().tolist()[0]
    print("finished gen<-------")

    return {"score": reward_score, "score+": reward_score_plus, "values": values, 
            "input_length": len(inputs.input_ids[0]), 
            "prompt_length": prompt_length, 
            "response_length": response_length
        }


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def generate_response(request: Request):
    data = await request.json()
    data = json.loads(json.dumps(data))
    print(data)
    history = data.get("history")
    predictRes = predict_reward(
        history= history 
    )
    
    return predictRes

if __name__ == '__main__':  
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-model-id", type=str, required=True)
    args = parser.parse_args()
    
    model_id = args.service_model_id
    

    port= SupportedModels.model_port[model_id]
    print("loading...", model_id, "\nport:", port)

    load_model(model_id=model_id)
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
