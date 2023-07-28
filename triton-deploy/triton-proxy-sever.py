import json
import uvicorn
from fastapi import FastAPI, HTTPException
import tritonclient.http as httpclient
import gevent.ssl
import numpy as np
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, AnyStr
import regex
import random
from contextlib import asynccontextmanager

class HttpClient:
    def __init__(self,
        host="localhost",port=8501,
        ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
        verbose=False):
        """

        :param url:
        :param ssl: Enable encrypted link to the server using HTTPS
        :param key_file: File holding client private key
        :param cert_file: File holding client certificate
        :param ca_certs: File holding ca certificate
        :param insecure: Use no peer verification in SSL communications. Use with caution
        :param verbose: Enable verbose output
        :return:
        """
        url = f"{host}:{port}"
        print("initializing-http from url=", url)

        if ssl:
            ssl_options = {}
            if key_file is not None:
                ssl_options['keyfile'] = key_file
            if cert_file is not None:
                ssl_options['certfile'] = cert_file
            if ca_certs is not None:
                ssl_options['ca_certs'] = ca_certs
            ssl_context_factory = None
            if insecure:
                ssl_context_factory = gevent.ssl._create_unverified_context
            triton_client = httpclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=True,
                ssl_options=ssl_options,
                insecure=insecure,
                ssl_context_factory=ssl_context_factory)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=url, verbose=verbose)

        self.triton_client = triton_client
    
    def infer(self, model_name,
          request_inputs:List[Dict[AnyStr, Any]], 
          request_outputs:List[AnyStr],
          request_compression_algorithm=None,
          response_compression_algorithm=None,
          streaming=False)-> httpclient.InferResult:
        """

        :param triton_client:
        :param model_name:
        :param request_compression_algorithm: Optional HTTP compression algorithm to use for the request body on client side.
                Currently supports "deflate", "gzip" and None. By default, no compression is used.
        :param response_compression_algorithm:
        :return:
        """
        
        inputs = []
        outputs = []
        # batch_size=4
        # 如果batch_size超过配置文件的max_batch_size，infer则会报错
        for item in request_inputs:
            name, data, dtype = item
            shape = data.shape
            inp =  httpclient.InferInput(
                name = name, shape= shape, datatype= dtype 
            )
            inp.set_data_from_numpy(data, binary_data=False)
            inputs.append(inp)
        
        for outname in request_outputs:
            out = httpclient.InferRequestedOutput(outname, binary_data=False)
            outputs.append(out)
        
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_compression_algorithm=request_compression_algorithm,
            response_compression_algorithm=response_compression_algorithm)
        # print(results)
        
        return results

class ModelWorker:
    model_id: str = None
    port = ""
    host=""
    system_prompt = ""
    bos_token = "<s>"
    eos_token= "</s>"
    roles : list = []


def build_prompt(query, history):
    prompt = [ ]
    if len(ModelWorker.roles)>=2:
        for (old_query, response) in history:
            prompt += [ModelWorker.roles[0] +": "+ old_query, ModelWorker.roles[1] +": "+ response  ]
        prompt += [ModelWorker.roles[0] +": "+query]
        prompt = "\n".join(prompt) +f"\n\n{ModelWorker.roles[1]}: "
    else:
        for (old_query, response) in history:
            prompt += [ old_query,  response  ]
        prompt += [query]
        prompt = "\n".join(prompt)
    
    add_system = len(prompt) < 500

    if add_system and ModelWorker.system_prompt :
        prompt = f"<s>{ModelWorker.system_prompt }{prompt}"
    else:
        prompt = f"<s>{prompt}"

    print(f"prompt:\n{prompt}\n******")

    return prompt

def str_as_bytes(text:str):
    return text.encode("utf-8")

def load_model(model_id=None):
    if ModelWorker.model_id is not None:
        return
    with open("configs/proxy.json") as f:
        
        config:dict = json.load(f)
        if model_id is None:
            model_id = config['selected']

        roles = config['roled_model_ids'][model_id]
        system_prompt = config['system'][model_id]
        model_port = config['port'][model_id]
        model_host = config['host'][model_id]
        service_port = config['service-port']
        

    ModelWorker.model_id = model_id
    ModelWorker.port = model_port
    ModelWorker.host = model_host
    ModelWorker.roles = roles
    ModelWorker.system_prompt = system_prompt
    print("loaded worker:", ModelWorker.__dict__)

    return service_port


async def predict_func(query, history, max_length, top_p, temperature, **kwargs ):
    print("kwargs:", kwargs)
    
    if len(ModelWorker.roles)>=2:
        role_reg= regex.compile(f"{ModelWorker.roles[0]}:|{ModelWorker.roles[1]}:")
    else:
        role_reg= None
    seed = kwargs.get('seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
    print('using random-seed:', seed)
    bos_id = 1
    eos_id = 2
    
    # input 0, 1, 2 is required
    prompt = build_prompt(query, history)
    bsz = 1
    input0 = [str_as_bytes(prompt)]
    input_list = [
        ("INPUT_0", np.array(input0).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("INPUT_1", np.array(bsz* [max_length]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("INPUT_2", np.array(bsz*[str_as_bytes("")]).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("INPUT_3", np.array(bsz*[str_as_bytes("")]).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("start_id", np.array(bsz*[bos_id]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("end_id", np.array(bsz*[eos_id]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("runtime_top_p", np.array(bsz*[top_p]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("temperature", np.array(bsz*[temperature]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("random_seed", np.array(bsz*[seed ]).astype(np.uint64).reshape(bsz, -1), "UINT64")
    ]
    output_list = [
        "OUTPUT_0"
    ]
    
    client = HttpClient(host=ModelWorker.host, port=ModelWorker.port)

    results=  client.infer(
        model_name= "ensemblebloom",
        request_inputs=input_list,
        request_outputs=output_list
    )
    output:str = results.get_output(output_list[0])['data'][0]
    print(output)
    index = output.index(ModelWorker.eos_token)
    output = output[len(prompt):index]
    if role_reg is not None:
        output = role_reg.sub("", output)
    print(type(output), output)
    return  output



@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    load_model()
    yield
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
from openai_api import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatMessage, DeltaMessage,
    EmbeddingUsageData,
    ChatRecRequest, ChatRecResponse
)


@app.post("/ask", response_model=ChatRecResponse)
async def response_request(request: ChatRecRequest):
    query = request.query

    response = await predict_func(
        query=query, history= [], max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
        seed = request.seed
    )
    
    return ChatRecResponse(response=response)

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

    response = await predict_func(
        query=query, history= history, max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
        seed = request.seed
    )
    
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, 
            choices=[choice_data], object="chat.completion",
            usage=EmbeddingUsageData())

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

    predictRes = await predict_func(
        query=query, history= history, max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
        seed = request.seed
    )        

    cur_str=""
    for new_response in [predictRes]:
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
    parser.add_argument("--service-port", type=int, required=False, default=None)
    args = parser.parse_args()
    
    model_id = args.service_model_id    
    model_port = args.service_port
    port= load_model(model_id)
    if model_port is not None:
        port = model_port
    print("loading...", model_id, "\nport:", port)

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
