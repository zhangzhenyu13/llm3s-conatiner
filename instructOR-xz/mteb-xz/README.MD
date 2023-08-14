# MTEB-XZ

## 介绍

METB-XZ 是基于[MTEB-zh](https://huggingface.co/moka-ai/m3e-base) 加入了很多小智特色leaderboard数据的评测框架，并且引入了simcse等模型作为评测基准，且加入了本地loading model方案，更佳直观高效。

MTEB-zh 是一个使用 [MTEB](https://github.com/embeddings-benchmark/mteb) 框架评测中文 Embedding 模型的 BenchMark，包含文本分类，文本重排，以及文本检索等任务。


## 已支持的模型
- [x] [simcse-xz] (our locally finetuned simcse based on simcse model)
- [x] [M3E](https://huggingface.co/moka-ai/m3e-base) (m3e-small, m3e-base)
- [x] [text2vec](https://github.com/shibing624/text2vec)
- [x] [DMetaSoul](https://huggingface.co/DMetaSoul/sbert-chinese-general-v2)
- [x] [UER](https://huggingface.co/uer/sbert-base-chinese-nli)
- [x] [ErLangShen](https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese)
- [x] [openai](https://openai.com/blog/new-and-improved-embedding-model)

## 评测

### 文本分类

- 小智业务数据集，包含4个hy分类器、一个通用分类器、3个qq数据、1个qr数据、senti、scene、xzfaq-quality数据等
- 数据集选择，选择开源在 HuggingFace 上的 6 种文本分类数据集，包括新闻、电商评论、股票评论、长文本等
- 评测方式，使用 MTEB 的方式进行评测，报告 Accuracy。


### 检索排序

#### T2Ranking 1W

- 数据集选择，使用 [T2Ranking](https://github.com/THUIR/T2Ranking/tree/main) 数据集，由于 T2Ranking 的数据集太大，openai 评测起来的时间成本和 api 费用有些高，所以我们只选择了 T2Ranking 中的前 10000 篇文章
- 评测方式，使用 MTEB 的方式进行评测，报告 map@1, map@10, mrr@1, mrr@10, ndcg@1, ndcg@10
- 注意！从实验结果和训练方式来看，除了 M3E 模型和 openai 模型外，其余模型都没有做检索任务的训练，所以结果仅供参考。



## 评测已支持的模型

1. 安装依赖
```bash
pip install -r requirements.txt
```
2. 运行openai类api模型的评测脚本
xz本地评测急哦啊笨
```bash
bash run_eval.sh
```

openai 类评测
```bash
python run_mteb_zh.py --model-type <model_type> --model-id <model_id: Optional>
```
