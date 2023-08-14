import os
import torch
import bisect
import random
from enum import Enum
from typing import Sized, cast, TypeAlias, Optional, Union
from dataclasses import dataclass, fields
from datasets import load_dataset, DatasetDict, load_from_disk
from datasets import arrow_dataset 
from torch.utils.data import Dataset, SequentialSampler

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
Tokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Union
)
base_str = ([chr(ord('a')+i) for i in range(26)] 
        #    +[chr(ord('A')+i) for i in range(26)]
)
def get_chr_r(num):
    res_str = ""
    if num ==0:
        return base_str[0]
    
    while num>0:
        idx = num% len(base_str)
        num= num// len(base_str)
        res_str+= base_str[idx]
    return res_str
                

class MultiTaskDataset:
    datasets: List[arrow_dataset.Dataset]
    cumulative_sizes: List[int]
    minSamples: int = int(1e+2) # do  not shard dataset smaller

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    def reset_epoch(self,):
        
        print("reseting epoch:", [self.seed, self._epoch, self.rank])
        random.seed(self.seed + self._epoch)
        self.datasets = []
        for ds in self.datasets_full:
            if self.num_replicas> 1 and len(ds)> self.minSamples:
                # make sure all process random state same to shards correctly
                self.datasets += [ds.shuffle(seed=self.seed + self._epoch)]
                curDsize = len(self.datasets[-1])
                self.datasets[-1] = \
                    self.datasets[-1].select(
                    range(self.rank,curDsize,self.num_replicas)
                )
                # print("added", len(self.datasets[-1]))
            else:
                # make sure different process shards differently in subprocess
                # this make small datasets upsampled
                self.datasets += [ds.shuffle(seed=self.seed + self._epoch* 10 + self.rank)]
        random.shuffle(self.datasets)
        self.cumulative_sizes: List[int] = self.cumsum(self.datasets)

        self._epoch += 1

    @property
    def column_names(self):
        return self.datasets_full[0].column_names
    
    def remove_columns(self, column_names: Union[str, List[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        for ds in self.datasets_full:
            ds.remove_columns(column_names=column_names, new_fingerprint=new_fingerprint)
        return self
    def __init__(self, datasets: Iterable[arrow_dataset.Dataset], rank=-1, world_size=1, seed=42) -> None:
        super().__init__()
        self.datasets_full = list(datasets)
        self.num_replicas: int = world_size
        self.rank:int = rank
        self.seed = seed
        self._epoch= 0
        self.reset_epoch()
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]

        

    def __len__(self):
        return self.cumulative_sizes[-1]


    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MultiTaskDistributedSampler(SequentialSampler):
    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)
    
    def __iter__(self) -> Iterator[int]:
        self.begin_epoch()
        return iter(range(len(self.data_source)))

    def begin_epoch(self,) -> None:
        # self.data_source
        self.data_source.reset_epoch()

    


class RecordType(str, Enum):
    PAIR = 'pair'
    TRIPLET = 'triplet'
    SCORED_PAIR = 'scored_pair'


@dataclass(slots=True)
class PairRecord:
    text: str
    text_pos: str


@dataclass(slots=True)
class TripletRecord:
    text: str
    text_pos: str
    text_neg: str


@dataclass(slots=True)
class ScoredPairRecord:
    sentence1: str
    sentence2: str
    label: float

@dataclass
class ConstrastiveCollator:
    tokenizer: Tokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def build_records(self, records, record_cls: PairRecord|TripletRecord|ScoredPairRecord):
        return [record_cls(**x) for x in records ]
    
    def pad(self, features):

        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

class PairCollator(ConstrastiveCollator):
    def __call__(self, records: list[PairRecord]) -> dict[str, torch.Tensor]:
        records = self.build_records(records, PairRecord)

        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = cast(torch.Tensor, text_pos_ids)

        features = {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
        }
        # res= self.pad(features=features)
        # print("---->/check", res)
        return features

class TripletCollator(ConstrastiveCollator):

    def __call__(self, records: list[TripletRecord]) -> dict[str, torch.Tensor]:
        records = self.build_records(records, TripletRecord)

        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]
        texts_neg = [record.text_neg for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_neg_ids = self.tokenizer(
            texts_neg,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']

        text_ids = cast(torch.Tensor, text_ids)
        text_pos_ids = cast(torch.Tensor, text_pos_ids)
        text_neg_ids = cast(torch.Tensor, text_neg_ids)
        features=  {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
            'text_neg_ids': text_neg_ids,
        }
        # return self.pad(features=features)
        return features

class ScoredPairCollator(ConstrastiveCollator):

    def __call__(self, records: list[ScoredPairRecord]) -> dict[str, torch.Tensor]:
        records = self.build_records(records, ScoredPairRecord)

        texts = [record.sentence1 for record in records]
        texts_pair = [record.sentence2 for record in records]
        labels = [record.label for record in records]
        labels = torch.tensor(labels, dtype=torch.float32)

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pair_ids = self.tokenizer(
            texts_pair,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pair_ids = cast(torch.Tensor, text_pair_ids)

        features = {
            'text_ids': text_ids,
            'text_pair_ids': text_pair_ids,
            'labels': labels,
        }
        
        return self.pad(features=features)
    
def load_dataset_local(dataset_id, verbose=True):
    localPath = os.path.join(os.environ['HOME'], "SharedData/instructOR")

    data= DatasetDict()
    if not isinstance(dataset_id, str):
        dataset_id = "/".join(dataset_id)
    data_path = os.path.join(localPath, dataset_id)
    for key in os.listdir(data_path):
        print("loading:", [dataset_id, key ] )
        data[key] = load_from_disk(os.path.join(data_path, key))
    print("loaded dataset:", dataset_id)
    if verbose:
        print(data)
    return data

if __name__ == "__main__":
    load_dataset_local(
        ('Hello-SimpleAI/HC3-Chinese', 'all')
    )
    load_dataset_local('cmrc2018')

