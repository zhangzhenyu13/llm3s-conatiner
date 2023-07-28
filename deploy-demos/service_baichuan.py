import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from typing import List
import regex
from configs.supported import SupportedModels
from streamerIter import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

class ModelWorker:
    model_id: str = None
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    streamer: TextIteratorStreamer = None
    roles : list = []

def load_model(model_id="baichuan-inc/baichuan-7B"):
    ModelWorker.model_id = model_id
    
    model_path = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    
    ModelWorker.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,)
    ModelWorker.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ModelWorker.streamer = TextIteratorStreamer(ModelWorker.tokenizer, skip_prompt=True)
    if model_id in SupportedModels.roled_model_ids:
        ModelWorker.roles = SupportedModels.roled_model_ids[model_id]
    print(ModelWorker.model)
    print(ModelWorker.tokenizer)
    print(ModelWorker.streamer)
    print("role info:", ModelWorker.roles)


def build_prompt(query, history= []):
    prompt = ""
    for q, r in history:
        prompt += q +"\n"
        prompt += r + "\n"
    prompt += query +"\n"
    return prompt


min_text_generation = 10
def predict_func(input, history, max_length, top_p, temperature, decoder="sample", **kwargs ):
    tokenizer, model, streamer= ModelWorker.tokenizer, ModelWorker.model, ModelWorker.streamer

    # input
    query = input.strip()
    sess_length = max_length

    role_reg= regex.compile(f"{ModelWorker.roles[0]}:|{ModelWorker.roles[1]}:")

    
    # ===== begin length process

    prompt = build_prompt(query=query, history=history)
    prompt = tokenizer.bos_token + prompt if tokenizer.bos_token!=None else prompt

    sess_length -= len(tokenizer.tokenize(prompt))
    if sess_length <= min_text_generation:
        print("max context length: ", [sess_length, max_length])
        raise StopIteration()


    print("Tokens for New Generation:", sess_length)
    print("EOS:",tokenizer.eos_token)
    print("additiobnal kwagrs:", kwargs, kwargs.get("repetition_penalty", 1.0))

    print(f"prompt:\n{prompt}\n******")
    # ===== finished length process
    
    inputs = tokenizer(prompt, return_tensors="pt")
    # print(inputs)

    if decoder == 'sample':
        print("using sample mode")
        generation_kwargs = dict(
            inputs= inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            min_new_tokens=min_text_generation,
            max_new_tokens=sess_length,
            early_stopping= True ,
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )
    elif decoder == 'contrastive':
        print("using contrastive mode")
        generation_kwargs = dict(
            inputs= inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            do_sample=False,
            penalty_alpha=0.6, 
            top_k=4,
            min_new_tokens=min_text_generation,
            max_new_tokens=sess_length,
            early_stopping= True ,
        )
    else:
        print("Warn: null decoder mode!!!")
        raise StopIteration()
    
    generation_kwargs['use_cache']= True # this feature is automatically enabled in transformers 4.28

    generation_kwargs['streamer'] = streamer

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    history += [(query, "")]
    for new_text in streamer:
        if not new_text:
            continue
        if new_text == prompt:
            continue
        generated_text = new_text
        if generated_text.endswith(tokenizer.eos_token):
            generated_text = generated_text.replace(tokenizer.eos_token, "")

        generated_text = generated_text.strip("�")
        yield generated_text
    thread.join()

    print("finished gen<-------")


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
from openai_api import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatMessage, DeltaMessage
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model != ModelWorker.model_id: 
        msg = f"wrong routing model-id:{request.model} --X-- {ModelWorker.model_id}"
        raise HTTPException(status_code=400, detail=msg)

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    if request.stream:
        generate = predict(query, history, request)
        return StreamingResponse(generate, media_type="text/event-stream")

    predictRes = predict_func(
        input=query, history= history, max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
    )
    
    for response in predictRes:
        ...
    print("gen data once", len(ModelWorker.tokenizer.tokenize(response)))
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

async def predict(query: str, history: List[List[str]], request: ChatCompletionRequest):
    model_id = ModelWorker.model_id
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    predictRes = predict_func(
        input=query, history= history, max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
    )        

    cur_str=""
    for new_response in predictRes:
        if len(new_response) == current_length:
            # print(f"cur/new:\n{cur_str}\n-----\n{new_response}")
            continue
        cur_str = new_response
        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))




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
