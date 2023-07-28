import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils import PaddingStrategy

class DataCollatorForRewardModel:
    model_input_names: List[str] = ["chosen_input_ids", "chosen_attention_mask", 'reject_input_ids', 'reject_attention_mask' ]

    def __init__(self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        padding_side: str = "right",
        truncation_side: str = "right",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        
        # init tokenizer
        # self.tokenizer.model_input_names = self.model_input_names
        self.tokenizer.padding_side = self.padding_side
        self.tokenizer.truncation_side = self.truncation_side


    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # print("f-type:",type(features))
        # if type(features) == dict:
        #     print("f-keys:",features.keys())
        # elif type(features) == list:
        #     print("list[0]:", features[0].keys(), len(features))

        chosen_input_ids = [f['chosen_input_ids'] for f in features]
        reject_input_ids = [f['reject_input_ids'] for f in features]
        chosen_attention_mask = [f['chosen_attention_mask'] for f in features]
        reject_attention_mask = [f['reject_attention_mask'] for f in features]
        # print("lens:",len(chosen_input_ids), len(reject_input_ids), len(chosen_input_ids[0]), len(reject_input_ids[0]))
        bsz= len(chosen_input_ids)

        input_ids = chosen_input_ids + reject_input_ids
        attention_mask = chosen_attention_mask + reject_attention_mask
        batch_features = {
            "input_ids": input_ids, "attention_mask": attention_mask
        }
        # print("batch-features:", batch_features)

        features = self.tokenizer.pad(
            batch_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        features = {}

        features['chosen_input_ids'] = input_ids[:bsz]
        features['reject_input_ids']= input_ids[bsz:]
        features['chosen_attention_mask'] = attention_mask[:bsz]
        features['reject_attention_mask'] = attention_mask[bsz:]

        # for k, v in features.items():
        #     if isinstance(v, torch.Tensor):
        #         print("collator:",k, v.size(), v)
        
        features = BatchEncoding(features, tensor_type=return_tensors)
        return features
