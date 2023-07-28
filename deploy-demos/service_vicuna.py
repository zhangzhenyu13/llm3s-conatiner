import torch
import os
import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, Literal, Optional, Union
import argparse
from configs.supported import SupportedModels
from transformers import AutoModel, AutoTokenizer
from utils import parse_text
from fastchat.model.model_adapter import load_model, add_model_args, get_conversation_template
from fastchat.conversation import Conversation
from fastchat.serve.inference import generate_stream
from fastchat.constants import ErrorCode,SERVER_ERROR_MSG

def get_model(base_model, args):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    model, tokenizer = load_model(
        base_model, args.device, args.num_gpus, args.max_gpu_memory, args.load_8bit, args.cpu_offloading
    )
    model = model.eval()
    
    print('finished loading', base_model)
    return model, tokenizer


class ModelWorker:
    model_id: str = None
    model: AutoModel = None
    tokenizer: AutoTokenizer = None
    with_role : bool = False
    role_info: list = []
    model_path: str = None


def initialze_model(model_id="vicuna/7b"):
    ModelWorker.model_id = model_id

    model_path = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    model, tokenizer = get_model(model_path, args)
    ModelWorker.model_path = model_path

    ModelWorker.model = model
    ModelWorker.tokenizer = tokenizer
    if model_id in SupportedModels.roled_model_ids:
        ModelWorker.with_role = True
        ModelWorker.role_info = SupportedModels.roled_model_ids[model_id]
    print(ModelWorker.model)
    print(ModelWorker.tokenizer)
    print("role info:", ModelWorker.with_role)


def generate_stream_gate(params, model, tokenizer, device, context_len=2048, stream_interval=2):
    print(params)
    try:
        for output in generate_stream(
            model,
            tokenizer,
            params,
            device,
            context_len = context_len,
            stream_interval= stream_interval,
        ):
            ret = {
                "text": output["text"],
                "error_code": 0,
            }
            if "usage" in output:
                ret["usage"] = output["usage"]
            if "finish_reason" in output:
                ret["finish_reason"] = output["finish_reason"]
            if "logprobs" in output:
                ret["logprobs"] = output["logprobs"]
            yield ret
    except torch.cuda.OutOfMemoryError as e:
        ret = {
            "text": f"{SERVER_ERROR_MSG}\n\n({e})",
            "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            "error_msg": "CUDA MEM Out"
        }
        yield ret
    except (ValueError, RuntimeError) as e:
        ret = {
            "text": f"{SERVER_ERROR_MSG}\n\n({e})",
            "error_code": ErrorCode.INTERNAL_ERROR,
            "error_msg": "Run Time ERROR"
        }
        # raise e
        yield ret


def build_prompt(query:str, history:list, conv: Conversation):
    for question, response in history:
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], response)
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def predict_func(input, history, max_length, top_p, temperature,**kwargs):
    tokenizer, model = ModelWorker.tokenizer, ModelWorker.model
    conv = get_conversation_template(ModelWorker.model_path)
    prompt = build_prompt(input, history, conv=conv)
    gen_params = {
            "model": ModelWorker.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": kwargs.get('repetition_penalty', 1.0),
            "max_new_tokens": max_length,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
    
    genRes = generate_stream_gate(gen_params, model, tokenizer, 
        args.device,
    )

    history.append((input, ""))
    for ret in genRes:
        # print(ret)
        response = ret['text']
        if ret['error_code'] != 0:
            history[-1] = (input, "")
            raise StopIteration()
        response = response.strip("�")
        yield response
        if ret['error_code'] != 0:
            break
    
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

    for new_response in predictRes:
        if len(new_response) == current_length:
            continue
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
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--service-model-id", type=str, required=True)
    args = parser.parse_args()
    
    model_id = args.service_model_id   
    port= SupportedModels.model_port[model_id]
    print("loading...", model_id, "\nport:", port)

    initialze_model(model_id=model_id)

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
