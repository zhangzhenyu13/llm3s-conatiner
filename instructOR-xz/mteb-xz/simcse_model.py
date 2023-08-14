import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Iterator
from numpy import ndarray
import tqdm


class SimCSE(object):
    max_bs= 256
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def half(self):
        self.model =  self.model.half()
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
                max_length: int = 512, **kwargs) -> Iterator[Union[ndarray, Tensor]]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        batch_size = min(self.max_bs, batch_size)
        
        if isinstance(sentence, str):
            sentence = [sentence]

        results = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm.tqdm(range(total_batch), desc=f'encoding({self.pooler})...'):
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
                
                embeddings =  embeddings.numpy().tolist()

                results.extend(embeddings)
        # if convert_to_numpy:
        #     results = np.array(results)
        return results
