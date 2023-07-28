import time
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, Literal, Optional, Union
import openai
from contextlib import asynccontextmanager

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "your-org"
    url: str = None
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelZoo:
    modelData:Dict[str, ModelCard] = dict()
    embedderData:Dict[str, ModelCard] = dict()

def load_models():
    from configs.supported import SupportedModels
    model_id2name=SupportedModels.model_names
    model_name2id={v:k for k, v in model_id2name.items() }
    ModelZoo.modelData.clear()
    for model_id in SupportedModels.model_ids:
        model_port = SupportedModels.model_port[model_id]
        model_host = SupportedModels.model_host[model_id]
        model= ModelCard(id= model_id, url=f"http://{model_host}:{model_port}/v1")
        ModelZoo.modelData[model_id] = model
    print("loaded generation models:", ModelZoo.modelData.keys() )

    for model_id in SupportedModels.embedder_names:
        model_port = SupportedModels.model_port[model_id]
        model_host = SupportedModels.model_host[model_id]
        model= ModelCard(id= model_id, url=f"http://{model_host}:{model_port}/v1")
        ModelZoo.embedderData[model_id] = model
    print("loaded embedder models:", ModelZoo.embedderData.keys() )

# define server
@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    load_models()
    yield
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


class ChatRecRequest(BaseModel):
    query: str
    decoder: str = "sample"
    repetition_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.85
    max_length: Optional[int] = 2048
    stream: Optional[bool] = False
    seed: Optional[int] = None
class ChatRecResponse(BaseModel):
    response: str


class EmbeddingCompletionRequest(BaseModel):
    model: str= "moka-ai/m3e-base"
    input: List[str]
    max_length: Optional[int] = 512
    encoding_format: str = 'base64'

class EmbeddingCompletionResponseData(BaseModel):
    object:str = "embedding"
    index: int
    embedding: List[float]
class EmbeddingUsageData(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
class EmbeddingCompletionResponse(BaseModel):
    object: str= "list"
    data: List[EmbeddingCompletionResponseData]
    model: str    
    usage: Optional[EmbeddingUsageData] 


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


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
    repetition_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.85
    max_length: Optional[int] = 2048
    stream: Optional[bool] = False
    seed: Optional[int] = None


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
    usage: Optional[EmbeddingUsageData] 

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=
        sorted(ModelZoo.modelData.values(), key=lambda x: x.id)
    )

@app.get("/v1/embedders", response_model=ModelList)
async def list_models():
    return ModelList(data=
        sorted(ModelZoo.embedderData.values(), key=lambda x: x.id)
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")

    messages = []
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    if request.stream:
        generate = predict(request, messages)
        return StreamingResponse(generate, media_type="text/event-stream")
    
    

    openai.api_base = ModelZoo.modelData[request.model].url
    openai.api_key = "none"
    response = openai.ChatCompletion.create(
            temperature=request.temperature, 
            top_p= request.top_p, 
            decoder= request.decoder,
            max_length = request.max_length, 
            repetition_penalty=request.repetition_penalty,
            model= request.model,
            messages=messages,
            stream=False
        )
    if hasattr(response.choices[0].message, "content"):
        # print("gen-->\n",response.choices[0].message.content)
        response = response.choices[0].message.content
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )
    else:
        raise HTTPException(status_code=401, detail="Worker Server Error")

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion",
            usage=EmbeddingUsageData() )


async def predict(request: ChatCompletionRequest, messages):
    openai.api_base = ModelZoo.modelData[request.model].url
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
            temperature=request.temperature, 
            top_p= request.top_p, 
            decoder= request.decoder,
            max_length = request.max_length, 
            repetition_penalty=request.repetition_penalty,
            model= request.model,
            messages=messages,
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                # print("gen--->", chunk.choices[0].delta.content)
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=chunk.choices[0].delta.content),
                    finish_reason=chunk.choices[0].finish_reason
                )
                chunk = ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk",
                        usage=EmbeddingUsageData())
                yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))


async def embeddings_func(request: EmbeddingCompletionRequest):
    openai.api_base = ModelZoo.embedderData[request.model].url
    openai.api_key = "none"
    response =  openai.Embedding.create(
        input = request.input, 
        model=request.model,
        max_length = request.max_length
    )
    results = []
    for x in response['data']:
        response = EmbeddingCompletionResponseData(
            index= x['index'],
            embedding=x['embedding']
        )
        results.append(response)
    return EmbeddingCompletionResponse(
        data= results, model=request.model, usage=EmbeddingUsageData()
    )

@app.post("/v1/embeddings", response_model=EmbeddingCompletionResponse)
async def generate_embeddings(request: EmbeddingCompletionRequest):
    return await embeddings_func(request)

@app.post("/v1/engines/your-org/embeddings", response_model=EmbeddingCompletionResponse)
async def generate_embeddings(request: EmbeddingCompletionRequest):
    return await embeddings_func(request)
