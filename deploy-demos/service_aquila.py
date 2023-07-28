import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, Literal, Optional, Union

from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.model.aquila_model import AQUILAModel
import bminf
from typing import List
import tqdm
from utils import Conversation, SeparatorStyle
class ModelWorker:
    model_id: str = None
    model: AQUILAModel = None
    tokenizer: Tokenizer = None


def load_model(model_id= 'aquilachat-7b' ):
    ModelWorker.model_id = model_id

    state_dict = os.path.join(os.environ['HOME'], "CommonModels/BAAI")

    loader = AutoLoader(
        "lm",
        model_dir=state_dict,
        model_name=model_id,
        use_cache=True)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    model.eval()
    model = model.half().cuda()

    ModelWorker.model = model
    ModelWorker.tokenizer = tokenizer
    print("finished loading model: ", [state_dict, model_id])


default_conversation = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    instruction="",
    roles=("Human", "Assistant", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def aquila_generate_stream(
        tokenizer: Tokenizer,
        model: AQUILAModel,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        prompts_tokens: List[List[int]] = None,
        stream_interval = 2,
    ) -> List[str]:
        # token_end_id depends
        # token_end_id = tokenizer.get_command_id('sep')
        stop_token_ids = [tokenizer.get_command_id("pad"), tokenizer.get_command_id("eos"),
                    tokenizer.get_command_id("sep") ]

        if prompts_tokens is not None:
            bsz = len(prompts_tokens)
            prompt_tokens = [torch.LongTensor(x) for x in prompts_tokens]
        else:
            bsz = len(prompts)
            prompt_tokens = [torch.LongTensor(tokenizer.encode(x)) for x in prompts]

        assert bsz == 1, "streaming mode only supports bsz=1."

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])


        total_len = min(2048, max_gen_len + max_prompt_size)

        output_ids = [] #list(prompt_tokens[0].cpu().numpy().tolist())

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = t.clone().detach().long() #torch.tensor(t).long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in tqdm.trange(start_pos, total_len):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)["logits"]
            #print(logits.shape)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            
            if next_token.item() in stop_token_ids:
                print("finished generate:", [start_pos, cur_pos, total_len, next_token.item(), tokenizer.decode([next_token.item()]) ])
                break
            
            output_ids.append(next_token.item())
            
            decoded_ = tokenizer.decode(output_ids)
            if ( len(output_ids) - max_prompt_size ) % stream_interval ==0:
                yield decoded_


            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded_ = tokenizer.decode(output_ids)
        yield decoded_

def build_prompt(query, history= []):
    conv = default_conversation.copy()
    for q, r in history:
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], r)

    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def predictCore(text, history, max_length, top_p, temperature):
    prompt = build_prompt(text, history=history)
    tokens = ModelWorker.tokenizer.encode_plus(prompt, None, max_length=None)['input_ids']
    print("prompt-encode:",tokens)
    tokens = tokens[1:-1]

    with torch.no_grad():
        out_iter = aquila_generate_stream(ModelWorker.tokenizer, ModelWorker.model, [text], 
                    max_gen_len=max_length, temperature=temperature, top_p=top_p,
                    prompts_tokens=[tokens])
        for text_piece in out_iter:
            # print(f"pred is {text_piece}")
            yield text_piece #.replace(prompt, "")

def predict_func(query, history, max_length, top_p, temperature,  **kwargs ):
    res = predictCore(text=query, history=history,max_length= max_length, top_p= top_p, temperature=temperature)
    for response in res:
        response = response.strip("�")
        yield response


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
        query=query, history= history, max_length=request.max_length, 
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
        query=query, history= history, max_length=request.max_length, 
        top_p=request.top_p, temperature=request.temperature,
        decoder=request.decoder,
        repetition_penalty = request.repetition_penalty,
    )        

    for new_response in predictRes:
        if len(new_response) == current_length:
            # print(f"cur/new:\n{cur_str}\n-----\n{new_response}")
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
    from configs.supported import SupportedModels
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-model-id", type=str, required=True)
    args = parser.parse_args()    
    model_id = args.service_model_id

    port= SupportedModels.model_port[model_id]
    print("loading...", model_id, "\nport:", port)

    load_model(model_id=model_id)
    for txt in predictCore("北京在哪里？", [], 200, 0.95, 0.8):
        print(txt)

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
