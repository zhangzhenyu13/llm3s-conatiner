import csv
import math
import sys
from collections import defaultdict
from enum import Enum
from typing import Iterable, TypeVar, cast
import os
from datasets import Dataset, DatasetDict, load_dataset
from mteb.abstasks import (
    AbsTaskClassification,
    AbsTaskPairClassification,
    AbsTaskReranking,
    AbsTaskRetrieval,
)
from tqdm import tqdm


T = TypeVar('T')
csv.field_size_limit(sys.maxsize)
  

class TaskType(str, Enum):
    Classification = 'Classification'
    Reranking = 'Reranking'
    Retrieval = 'Retrieval'
    PairClassification = 'PairClassification'
    All = 'All'

taskFolder= os.path.join(os.environ['HOME'], "SharedData/mteb_xz")

class TaskInstructLoader:
    def __init__(self) -> None:
        self.insts = None
        self.task_type = None
    def set_instructions(self, insts, task_type):
        self.insts = insts
        self.task_type= task_type
    
    def inst_inject(self, dataset:Dataset):
        try:
            if self.insts is None:
                return dataset
        except Exception as e:
            print(e.args)
            return dataset

        if self.task_type == TaskType.Classification:
            dataset = dataset.rename_columns({"text": "s"}).map(
                lambda x: {"text": self.insts[0]+": "+ x['s']}
                ).remove_columns("s")
            return dataset
        elif self.task_type == TaskType.PairClassification:
            # print("qq-insts:", self.insts, dataset)
            def process_inst(x):
                # print([self.insts, x])
                # print(self.insts[0], len(x), len(x['s1']) )
                if len(self.insts)==1:
                    return {
                        "sent1": [self.insts[0]+": "+ txt for txt in x['s1'] ],
                        "sent2": [self.insts[0]+": "+ txt for txt in x['s2'] ],
                        }
                else:
                    return {
                    "sent1": [self.insts[0]+": "+ txt for txt in x['s1'] ],
                    "sent2": [self.insts[1]+": "+ txt for txt in x['s2'] ],
                    }
            dataset = dataset.rename_columns({"sent1": "s1", "sent2":"s2"}
            ).map(process_inst).remove_columns(["s1", "s2"])
            return dataset
        elif self.task_type == TaskType.Retrieval:
            corpus, queries, qrels = dataset 
            # print("corpus", list(corpus.items())[:10])
            # print("queries:", list(queries.items())[:10])
            # print("qrels:", list(qrels.items())[:10])
            def inst_corpus(items, inst):
                res_txts = {}
                for qid, cor in items:
                    txt = cor['text']
                    res_txts[qid] = {"text": inst+": "+ txt}
                return res_txts
            def inst_queries(items, inst):
                return {qid: inst+": "+ txt for qid, txt in items}
            
            corpus = inst_corpus(corpus.items(), self.insts[1])
            queries = inst_queries(queries.items(), self.insts[0])
            return corpus, queries, qrels
        
        elif self.task_type == TaskType.Reranking:
            dataset.rename_columns(
                {"query":"q", "positive":"pos", "negative":"neg"}
                ).map(lambda x:{
                    "query": self.insts[0]+": "+ x['q'],
                    "positive": self.insts[1]+": "+ x['pos'],
                    "negative": self.insts[1]+": "+ x['neg']
                }).remove_columns(['q','pos', 'neg'])
            return dataset
        else:
            raise ValueError(f"not supported task type: {self.task_type}")
        



class XZFaqQuality(AbsTaskClassification, TaskInstructLoader):
    
    @property
    def description(self):
        return {
            'name': 'xzfaq-quality',
            'hf_hub_name': '',
            'description': 'xzfaq iflytek',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
            'n_experiments': 3,
        }

    def load_data(self, **kwargs):
        dataPath = {
            "train": os.path.join(taskFolder, self.description['name'], 'train.json'),
            "test": os.path.join(taskFolder, self.description['name'], 'test.json'),
        }
        dataset = load_dataset('json', data_files=dataPath)
        dataset = dataset.rename_column('sentence', 'text')
        dataset = self.inst_inject(dataset)
        self.dataset = dataset
        
        self.data_loaded = True


class XZHyClassifier(AbsTaskClassification, TaskInstructLoader):
    def __init__(self, method="logReg", n_experiments=None, samples_per_label=None, k=3, batch_size=32, **kwargs):
        print(kwargs)
        self.taskname = kwargs.get('taskname')
        super().__init__(method, n_experiments, samples_per_label, k, batch_size, **kwargs)
        

    @property
    def description(self):
        return {
            'name': self.taskname,
            'hf_hub_name': '',
            'description': f'xz-hy-classifier {self.taskname}',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
            'n_experiments': 3,
        }

    def load_data(self, **kwargs):
        dataPath = {
            "train": os.path.join(taskFolder, self.description['name'], 'train.json'),
            "test": os.path.join(taskFolder, self.description['name'], 'test.json'),
        }
        dataset = load_dataset('json', data_files=dataPath)
        dataset = dataset.rename_column('sentence', 'text')
        dataset = self.inst_inject(dataset)
        self.dataset = dataset
        self.data_loaded = True


class XZClassifier(AbsTaskClassification, TaskInstructLoader):
    def __init__(self, method="logReg", n_experiments=None, samples_per_label=None, k=3, batch_size=32, **kwargs):
        print(kwargs)
        self.taskname = kwargs.get('taskname')
        super().__init__(method, n_experiments, samples_per_label, k, batch_size, **kwargs)
        

    @property
    def description(self):
        return {
            'name': self.taskname,
            'hf_hub_name': '',
            'description': f'xz-general-classifier {self.taskname}',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
            'n_experiments': 3,
        }

    def load_data(self, **kwargs):
        dataPath = {
            "train": os.path.join(taskFolder, self.description['name'], 'train.json'),
            "test": os.path.join(taskFolder, self.description['name'], 'test.json'),
        }
        dataset = load_dataset('json', data_files=dataPath)
        def process_input(datapoint):
            # print(datapoint)
            s1, s2 = datapoint['sentence1'], datapoint['sentence2']
            return {
                "text": s1+ s2, "label": datapoint['label']
            }
        dataset = dataset.map(lambda x: process_input(x))
        dataset = self.inst_inject(dataset)
        self.dataset = dataset
        self.data_loaded = True


class XZQQPairs(AbsTaskPairClassification, TaskInstructLoader):
    def __init__(self, **kwargs):
        print(kwargs)
        self.taskname = kwargs.get('taskname')
        super().__init__(**kwargs)

    @property
    def description(self):
        return {
            'name': self.taskname,
            'hf_hub_name': f'xz-qq/qr task: {self.taskname}',
            'category': 's2s',
            'type': 'PairClassification',
            'eval_splits': ['train'],
            'eval_langs': ['zh'],
            'main_score': 'ap',
        }

    def load_data(self, **kwargs):
        def transform_label(label):
            if label in ("good", "bad"):
                return 1 if label =='good' else 0
            if label in ("0", "1"):
                return 1 if label =='1' else 0
            if label in (0, 1):
                return label
            raise ValueError(f"label-format: {label}")
        
        # dataset = load_dataset('vegaviazhang/Med_QQpairs')['train']  # type: ignore
        dataset = load_dataset("json",
            data_files={"train": os.path.join(taskFolder, self.taskname, 'train.json'),
            #   "test": os.path.join(taskFolder, self.taskname, 'test.json') 
              }
            )
        if False:
            def process(item):
                item = cast(dict, item)
                x={}
                x['sent1'] = item['sentence1']
                x['sent2'] = item['sentence2']
                x['labels'] = transform_label(item['label'])
                return x
            
            self.dataset = dataset.map(process)
        else:
            dataset = dataset['train']
            record = {'sent1': [], 'sent2': [], 'labels': []}
            for item in dataset:
                item = cast(dict, item)
                record['sent1'].append(item['sentence1'])
                record['sent2'].append(item['sentence2'])
                record['labels'].append(transform_label(item['label']))
            dataset = DatasetDict(train=Dataset.from_list([record]))
        dataset = self.inst_inject(dataset)
        self.dataset= dataset
        print(self.dataset)
        self.data_loaded = True