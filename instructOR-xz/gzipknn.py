import gzip
import numpy as np
import collections
import tqdm
import json
import os
import functools
from multiprocessing import Pool 
from typing import List, Tuple, AnyStr



def compress_core(text:str):
    return len(gzip.compress(text.encode()))

def compute_ncd_class(x1, K, train_set: List[Tuple[AnyStr, AnyStr]]):
    distance_x1 = []
    Cx1 = compress_core(x1)
    for x2, _ in train_set:
        Cx2 = compress_core(x2)
        x1x2:str = " ".join([x1, x2])
        Cx1x2 = compress_core(x1x2)
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_x1.append(ncd)
    sorted_idx = np.argsort(np.array(distance_x1))
    topk_class = collections.Counter(
        [train_set[idx][1] for idx in sorted_idx[:K]]
    )
    # print(topk_class)
    predict_class = max(topk_class.items(), key=lambda x: x[1])

    return predict_class[0]

def textknn(test_set: List[Tuple[AnyStr, AnyStr]], train_set: List[Tuple[AnyStr, AnyStr]], K:int):
    results = []
    
    batchSize = 1000
    testStrList = [x[0] for x in (test_set)]
    trainStrList = [x[0] for x in (train_set)]
    test_train_StrList = [" ".join([x1,x2]) for x1,_ in test_set for x2,_ in train_set]
    with Pool(40) as workers:
        def batch_func(datalist):
            batched = [datalist[i:i+batchSize] for i in range(0, len(datalist), batchSize)]
            cres = []
            for batch in tqdm.tqdm(batched):
                cres += workers.map(compress_core, batch)
            return cres
        print("test gzip")
        
        testCX = batch_func(testStrList)
        print("train gzip")
        trainCX = batch_func(trainStrList)
        print("test-train gzip")
        test_train_CX = batch_func(test_train_StrList)
    dictCX= dict(zip(
        testStrList + trainStrList + test_train_StrList,
        testCX + trainCX + test_train_CX
    ))

    for x1, _ in tqdm.tqdm(test_set):
        Cx1 = dictCX[x1]
        distance_x1 = []
        for x2, _ in train_set:
            Cx2 = dictCX[x2]
            x1x2:str = " ".join([x1, x2])
            Cx1x2 = dictCX[x1x2]
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_x1.append(ncd)
        sorted_idx = np.argsort(np.array(distance_x1))
        topk_class = collections.Counter(
            [train_set[idx][1] for idx in sorted_idx[:K]]
        )
        predict_class = max(topk_class.items(), key=lambda x: x[1])
        results.append(predict_class)
    return results

def textknnV2(test_set: List[Tuple[AnyStr, AnyStr]], train_set: List[Tuple[AnyStr, AnyStr]], K:int):
    results = []
    
    batchSize = 40
    testStrList = [x[0] for x in (test_set)] #[:10]
    with Pool(batchSize) as workers:
        process_func = functools.partial(compute_ncd_class,
            K=K, train_set= train_set )
        
        def batch_func(datalist):
            batched = [datalist[i:i+batchSize] for i in range(0, len(datalist), batchSize)]
            cres = []
            for batch in tqdm.tqdm(batched):
                cres += workers.map(process_func, batch)
            return cres
        print("test gzip")
        results = batch_func(testStrList)
    
    return results

def load_sentence_class(filename):
    data = []
    with open(filename) as f:
        for line in tqdm.tqdm(f):
            x= json.loads(line)
            data.append((x['sentence'], x['label']))
    
    return data

def compute_metrics(labels, preds, labelMapper=None):
    assert len(labels) == len(preds)
    if len(labels) > len(preds):
        labels = labels[:len(preds)]
    print(f"metrics, size data/lable = {len(labels)}/{len(labelMapper)}")
    if labelMapper is not None:
        labels = [labelMapper[l] for l in labels]
        preds = [labelMapper[l] for l in preds]

    from sklearn.metrics import f1_score, accuracy_score

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    print("acc/f1=", acc, f1)


if __name__ == '__main__':
    task= "hy-dn"
    #hy-bx/   hy-cfxd/ hy-dn/   hy-hfp/  hy-kt/   hy-xyj/ 
    dataPath = os.path.join(os.environ['HOME'], f'SharedData/mteb_xz/{task}')
    testfilename = os.path.join(dataPath, "test.json")
    trainfilename = os.path.join(dataPath, "train-s.json")

    testdata , traindata = load_sentence_class(testfilename), load_sentence_class(trainfilename)

    labels = [x[1] for x in testdata]
    labelMapper = {l: idx for idx, l in
                    enumerate(set(map(lambda x: x[1], traindata)))}
    # print(labelMapper)
    predicts = textknnV2(testdata, traindata, K=5)


    compute_metrics(labels, predicts, labelMapper)
