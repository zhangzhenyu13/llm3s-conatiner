from typing import Tuple, List
import tqdm
from datasets import DatasetDict, Dataset
import faiss
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(
    os.path.dirname(
        os.path.abspath(sys.argv[0])
    )
))
from dataloader import load_dataset_local
from embedder import ContrastiveEmbedder
from transformers import AutoTokenizer

from transform_datasets import save_ds_as_json, ds_map_func

def load_qcls(dataset_id='gcls'):
    def concate_s1_2(x):
        s1, s2 = x['sentence1'], x['sentence2']
        s1= s1.split("\t")
        label = x['r']
        return {
            "sentence": "\t".join(s1+ [s2]),
            "label": label
        }
    
    dataset_dict = load_dataset_local(dataset_id)
    dataset_dict = dataset_dict.rename_column("label", "r")
    dataset_dict= ds_map_func(dataset_dict, concate_s1_2
            ).select_columns(["sentence", "label"])
    
    return dataset_dict

def get_sentence_encoder(modelpath=None, pooling_strategy=None
    )->Tuple[ContrastiveEmbedder, AutoTokenizer]:
    from embedder import ContrastiveEmbedder
    from transformers import AutoTokenizer
    if modelpath is None:
        modelpath = os.path.join(os.environ['HOME'], "CommonModels/moka-ai/m3e-base")

        model = ContrastiveEmbedder.from_pretrained(model_name_or_path=modelpath,
                                                pooling_strategy='last_mean')
    else:
        model = ContrastiveEmbedder.from_pretrained(model_name_or_path=modelpath,
                                    pooling_strategy=pooling_strategy)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    return model, tokenizer

def mine_hard_classification(model:ContrastiveEmbedder, 
                  tokenizer:AutoTokenizer, 
                  ds:Dataset):
    texts, labels = ds['sentence'], ds['label']
    # find a text similar to txt1 with label=0
    # fina a text similar to txt2 with label=0
    sampleSize = len(texts)

    embs_list = model.batch_encode(texts, tokenizer, batch_size=batch_size, max_length=max_length)
    
    model_d = embs_list[0].shape[1]
    nlist= 100
    nbits= 16
    M=128

    quantizer = faiss.IndexFlatIP(model_d)
    indexFlat = faiss.IndexIVFFlat(quantizer, model_d, nlist, faiss.METRIC_INNER_PRODUCT)
    indexFlat.train(np.concatenate(embs_list, axis=0))

    indexFlat = faiss.index_cpu_to_gpus_list( indexFlat, ngpu=1 )
    
    [ indexFlat.add(embs) for embs in  embs_list ]

    indexFlat.nprobe = 100

    print(indexFlat.ntotal, sampleSize)

    samples = []
    found= set()
    searchK= min(2048, 100*batch_size)
    for idx_batch, embs in tqdm.tqdm(enumerate(embs_list) ) :
        cosines, indexes = indexFlat.search(embs, k=searchK)
        # print(indexes)
        for i in range(len(indexes)):
            curIndex = idx_batch* batch_size + i

            index, sims = indexes[i], cosines[i]
            
            def get_hard_neg(index, sims):
                for idx, sim in zip(index, sims):
                    # print('info=',idx, sim)
                    if idx == curIndex:
                        continue  
                    if labels[idx] != labels[curIndex]:
                        return (idx, sim)
                return None, None
            def get_positive(index, sims):
                for idx, sim in zip(index, sims):
                    if idx == curIndex:
                        continue
                    if labels[idx] == labels[curIndex]:
                        return (idx, sim)
                return None, None
            # fetch twice
            pos_idx, sim_pos = get_positive(index=index, sims=sims)
            neg_idx, sim_neg = get_hard_neg(index=index, sims=sims)
            if pos_idx is None or neg_idx is None:
                continue
            # print([pos_idx, neg_idx, sim_pos, sim_neg])
            txt = texts[curIndex]
            txt_pos = texts[pos_idx]
            txt_neg = texts[neg_idx]
            key = txt_pos + txt
            if key in found:
                continue
            found.add(key)
            key2 = txt + txt_pos
            found.add(key2)

            samples.append({
                "text": txt, 
                "text_pos": txt_pos,
                "text_neg": txt_neg,
                }
            )
            
    return samples


def mine_hard_neg_qq(model:ContrastiveEmbedder, 
                  tokenizer:AutoTokenizer, 
                  ds:Dataset):
    texts1, texts2, labels = ds['text'], ds['text_pos'], ds['label']
    # find a text similar to txt1 with label=0
    # fina a text similar to txt2 with label=0
    sampleSize = len(texts1)
    embs_list1 = model.batch_encode(texts1, tokenizer, batch_size=batch_size, max_length=max_length)
    embs_list2 = model.batch_encode(texts2, tokenizer, batch_size=batch_size, max_length=max_length)
    
    indexFlat = faiss.IndexFlatIP(embs_list1[0].shape[1])
    [ indexFlat.add(embs) for embs in embs_list1 + embs_list2 ]

    print(indexFlat.ntotal, sampleSize)

    samples = []
    searchK= min(2048, 100*batch_size)
    for idx_batch, (embs, embs2) in tqdm.tqdm(enumerate(zip(embs_list1, embs_list2)) ) :
        cosines, indexes = indexFlat.search(embs, k=searchK)
        cosines2, indexes2 = indexFlat.search(embs2, k=searchK)

        for i in range(len(indexes)):
            curIndex = idx_batch* batch_size + i
            if labels[curIndex] != 1:
                continue
            # print([labels[curIndex], texts1[curIndex], texts2[curIndex]  ])
            # print(ds[curIndex])
            # input()
            index1, index2 = indexes[i], indexes2[i]
            sims1, sims2 = cosines[i], cosines2[i]
            top1_item= []
            def get_hard_neg(index, sims):
                for idx, sim in zip(index, sims):
                    # print('info=',idx, sim)
                    if idx == curIndex:
                        continue
                    tag= True
                    if idx >= sampleSize:
                        idx -= sampleSize
                        tag= False
                    if labels[idx] == 0:
                        return (idx, sim, tag)
            # fetch twice and merge     
            if (res:=get_hard_neg(index1, sims1) ) is not None:
                top1_item.append(res)
            if (res:=get_hard_neg(index2, sims2) ) is not None:
                top1_item.append(res)
            if len(top1_item) == 0:
                continue
            if len(top1_item)>1:
                neg_idx, neg_sim, is_txt1 = top1_item[0] if top1_item[0][1]> top1_item[1][1] else top1_item[1]
            else:
                neg_idx, neg_sim, is_txt1 = top1_item[0]
            txt_neg = texts1[neg_idx] if is_txt1 else texts2[neg_idx]
            samples.append({
                "text": texts1[curIndex], 
                "text_pos": texts2[curIndex],
                "text_neg": txt_neg,
                }
            )
            # print(samples[-1])
            # print("ext:", [texts1[neg_idx], texts2[neg_idx], labels[neg_idx], neg_sim])
    
    return samples
    

def mine_hard_neg_asymmetric(model:ContrastiveEmbedder, 
                  tokenizer:AutoTokenizer, 
                  ds:Dataset):
    texts1, texts2 = ds['text'], ds['text_pos']
    embs_list1 = model.batch_encode(texts1, tokenizer, batch_size=batch_size, max_length=max_length)
    embs_list2 = model.batch_encode(texts2, tokenizer, batch_size=batch_size, max_length=max_length)
    
    indexFlat1 = faiss.IndexFlatIP(embs_list1[0].shape[1])
    [ indexFlat1.add(embs) for embs in embs_list1 ]

    indexFlat2 = faiss.IndexFlatIP(embs_list1[0].shape[1])
    [ indexFlat2.add(embs) for embs in embs_list2]
    
    print(indexFlat1.ntotal,indexFlat2.ntotal, len(embs_list1))

    samples = []
    searchK= min(2048, 100*batch_size)
    for idx_batch, (embs, embs2) in tqdm.tqdm(enumerate(zip(embs_list1, embs_list2)) ) :
        cosines, indexes = indexFlat1.search(embs, k=searchK)
        cosines2, indexes2 = indexFlat2.search(embs2, k=searchK)


        neg_scores = []
        pos_scores = []
        for i in range(len(cosines)):
            curIndex = idx_batch* batch_size + i
            index1, index2 = indexes[i], indexes2[i]
            sim1, sim2 = cosines[i], cosines2[i]
            # print("all", len(sim1))
            common= set(index1).intersection(index2)
            # print("com1", len(common))
            common = common.difference([curIndex])
            # print("com2", len(common))

            if len(common) <1 :
                continue
            data1 = dict(zip(index1, sim1))
            data2 = dict(zip(index2, sim2))
            
            for idx in common:
                s1 = data1[idx]
                s2 = data2[idx]
                pos_scores.append((s1+s2, idx) ) # txt-i and txt-j is similar and out-i and out-j is similar 
                neg_scores.append((s1-s2, idx) ) # txt-i and txt-j is similar but out-i and out-j is not

            # pos_pair = max(pos_scores, key=lambda x: x[0])
            neg_pair = max(neg_scores, key=lambda x: x[0])
            # print(pos_pair, neg_pair)
            samples.append({
                "text": texts1[curIndex], 
                "text_pos": texts2[curIndex],
                "text_neg": texts2[neg_pair[1]],
                }
            )
            # print(samples[-1])
    
    return samples



if __name__ == "__main__":
    transform_triplets= False
    batch_size = 128
    max_length = 512

    # modelPath= os.path.join(os.environ['HOME'], "CommonModels/simcse/chinese_simbert_L-12_H-768_A-12")
    modelPath= os.path.join(os.environ['HOME'], "CommonModels/moka-ai/m3e-small")

    model, tokenizer = get_sentence_encoder(modelpath=modelPath,pooling_strategy="last_mean")

    model= model.cuda()

    dsPart='train'

    task_mapper={
        "xzfaq-quality": load_dataset_local,
        "hy-bx": load_dataset_local,
        "hy-dn": load_dataset_local,
        "hy-hfp": load_dataset_local,
        "hy-kt": load_dataset_local,
        "hy-sjpj": load_dataset_local,
        "hy-yzrsq": load_dataset_local,
        "hy-xyj": load_dataset_local,
        "hy-cfxd": load_dataset_local,
        "hy-dspb": load_dataset_local,
        "hy-nfyyp": load_dataset_local,
        "senti": load_dataset_local,
        "scene": load_dataset_local,
        "gcls": load_qcls,

        "clue/iflytek": load_dataset_local,
        "clue/tnews": load_dataset_local,

    }
    
    for task in task_mapper:
        loader_func = task_mapper[task]
        ds= loader_func(dataset_id= task)
        samples = mine_hard_classification(model=model, tokenizer=tokenizer,ds=ds[dsPart])
        ds = DatasetDict(**{dsPart:Dataset.from_list(samples)})

        save_ds_as_json(
            ds,
            "triplets",
            task
        )
