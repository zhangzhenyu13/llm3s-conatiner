
# Visit http://localhost:port/docs for documents.

import os
import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel

def get_model(base_model):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True).half().cuda()
    model = model.eval()
    print('finished loading', base_model)
    return model

class ModelWorker:
    model_id: str = None
    model: AutoModel = None
    tokenizer: AutoTokenizer = None

def load_model(model_id="THUDM/chatglm2-6b"):
    model_path = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    ModelWorker.model_id = model_id
    ModelWorker.model = get_model(model_path)
    ModelWorker.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(ModelWorker.model)
    print(ModelWorker.tokenizer)

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)



class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    decoder: str = "sample"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model != ModelWorker.model_id: 
        msg = f"wrong routing model-id:{request.model} --X-- {ModelWorker.model_id}"
        raise HTTPException(status_code=400, detail=msg)
    model, tokenizer = ModelWorker.model, ModelWorker.tokenizer

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
        generate = predict(query, history, request.model)
        return StreamingResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(query: str, history: List[List[str]], model_id: str):
    model, tokenizer = ModelWorker.model, ModelWorker.tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
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


if __name__ == "__main__":
    from configs.supported import SupportedModels
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-model-id", type=str, required=True)
    args = parser.parse_args()
    
    model_id = args.service_model_id
    
    port= SupportedModels.model_port[model_id]
    print("loading...", model_id, "\nport:", port)
    load_model(model_id=model_id)

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)