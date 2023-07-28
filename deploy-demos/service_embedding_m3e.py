from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Union, Iterator
import os
import torch
from configs.supported import SupportedModels
from transformers import AutoTokenizer, AutoModel
import tqdm
from openai_api import (
    EmbeddingUsageData, EmbeddingCompletionRequest,
    EmbeddingCompletionResponse, EmbeddingCompletionResponseData
)

if torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("Not Supported")

class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
            
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None

        if pooler is not None:
            print(f"Creating a new pooler with {pooler} pooling.")
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            print("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            print("Creating cls pooling one.")
            self.pooler = "cls"
    
    def encode(self, sentence: Union[str, List[str]], 
                batch_size: int = 64,
                device: str = None, 
                normalize_to_unit: bool = True,
                keepdim: bool = True,
                max_length: int = 128) -> Iterator[List]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        results = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm.tqdm(range(total_batch)):
                batch_sentence = sentence[batch_id*batch_size:(batch_id+1)*batch_size]
                inputs = self.tokenizer(
                    batch_sentence, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "mean":
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    embeddings = (hidden_states * attention_mask.unsqueeze(2)).sum(1) / (1e-6 + attention_mask.sum(1)).unsqueeze(1)
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embeddings = embeddings.cpu()
                if single_sentence and not keepdim:
                  embeddings = embeddings[0]
                
                embeddings =  embeddings.numpy().tolist()

                results += embeddings
                
        return results


class ModelWorker:
    model_id: str = None
    model: SimCSE = None

def load_model(model_id="moka-ai/m3e-base"):
    ModelWorker.model_id = model_id
    model_path = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    ModelWorker.model = SimCSE(model_path,
        device=device, pooler='mean')
    print(ModelWorker.model)


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)





@app.post("/v1/embeddings", response_model=EmbeddingCompletionResponse)
async def generate_embeddings(request: EmbeddingCompletionRequest):
    if request.model != ModelWorker.model_id: 
        msg = f"wrong routing model-id:{request.model} --X-- {ModelWorker.model_id}"
        raise HTTPException(status_code=400, detail=msg)

    texts = request.input
    max_length = request.max_length
    embeddings = ModelWorker.model.encode(texts, max_length=max_length)
    results = []
    for emb in embeddings:
        response = EmbeddingCompletionResponseData(
            index= len(results),
            embedding=emb
        )
        results.append(response)
    return EmbeddingCompletionResponse(
        data= results, model=request.model, usage=EmbeddingUsageData()
    )

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
