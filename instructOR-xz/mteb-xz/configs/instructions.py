instructionsMapper={
"flagretriver": ['为这个句子生成表示以用于检索相关文章：', ''],
"qd-retriever":["表征这个问题以支持相关文档检索", "表征这个文档以支持检索"],
"wd-retriever":["表征这些关键词以支持相关文档检索", "表征这个文档以支持关键词检索"],
"qq-retriever":["表征这个问题以支持重复问题检索"],
"qr-retriever":["表征这个问题以支持相关回复检索", "表征这个回复以支持问答检索"],
"cr-retriever":["表征这个对话上下文以支持对话检索", "表征这个对话回复以支持检索"],
"cls-retriever": ["表征这个问题以支持同类问题检索"],

"cmrc2018": ["表征这个问题以支持相关文档检索", "表征这个文档以支持检索"],
"vegaviazhang/Med_QQpairs": ["表征这个医学问题以支持重复问题检索"],
"shibing624/nli_zh/ATEC": ["表征这个问题以支持重复问题检索"],
"shibing624/nli_zh/BQ": ["表征这个问题以支持重复问题检索"],
"shibing624/nli_zh/LCQMC": ["表征这个问题以支持重复问题检索"],
"shibing624/nli_zh/PAWSX": ["表征这个问题以支持重复问题检索"],
"shibing624/nli_zh/STS-B": ["表征这个问题以支持重复问题检索"],
"shibing624/sts-sohu2021": ["表征这个问题以支持主要含义相同问题检索"],
"shibing624/snli-zh":["表征这个问题以支持重复问题检索"],
"wangrui6/Zhihu-KOL": ["表征这个问答网站的问题以支持答案检索", "表征这个问答网站的答案以支持检索"],
"Hello-SimpleAI/HC3-Chinese/all": ["表征这个问题以支持答案检索", "表征这个答案以支持检索"],
"wiki_atomic_edits/chinese_insertions":["表征这个文档以支持相似文档检索"],
"wiki_atomic_edits/chinese_deletions": ["表征这个文档以支持相似文档检索"],
"michaelwzhu/ChatMed_Consult_Dataset": ["表征这个医疗问诊的问题以支持问诊结果检索", "表征这个医疗问诊结果以支持检索"],
"michaelwzhu/ShenNong_TCM_Dataset": ["表征这个医疗问诊的问题以支持问诊结果检索", "表征这个医疗问诊结果以支持检索"],
"amazon_reviews_multi/zh": ["表征这个评论标题以支相关持评论内容检索", "表征这个评论内容以支持检索"],
"csebuetnlp/xlsum/chinese_simplified": ["表征这个文章标题以支持相关文档检索", "表征这个文档内容以支持检索"],
"mlqa/mlqa-translate-train.zh": ["表征这个问题以支持相关文档检索", "表征这个文档以支持检索"],
"clue/afqmc": ["表征这个问题以支持重复问题检索"],
"clue/c3": ["表征这个问题以支持相关文档/对话内容检索", "表征这个文档/对话以支持检索"],
"clue/chid": ["表征这些关键词以支持相关文档检索", "表征这个文档以支持关键词检索"],
"clue/cmnli": ["表征这个问题以支持重复问题检索"],
"clue/csl": ["表征这些关键词以支持相关文档检索", "表征这个文档以支持关键词检索"],
"clue/drcd": ["表征这个问题以支持相关文档检索", "表征这个文档以支持检索"],
"clue/iflytek": ["表征这个问题以支持同类问题检索"],
"clue/ocnli": ["表征这个问题以支持同类问题检索"],
"clue/tnews": ["表征这个问题以支持同类问题检索"],
"suolyer/webqa": ["表征这个知识问答的问题以支持答案检索", "表征这个答案以支持检索"],
"neuclir/csl": ["表征这个{CATE}领域文档的标题/关键词以支持相关文档检索", "表征这个{CATE}领域的文档以支持检索"],
"PaddlePaddle/dureader_robust": ["表征这个问题以支持相关文档检索", "表征这个文档以支持检索"], 
"miracl/miracl-corpus/zh": ["表征这个文章标题以支持文档检索", "表征这个文档内容以支持检索"],
# xz dataset
"xzqq": ["表征这个电商领域的问题以支持重复问题检索"],
"kfqq": ["表征这个问题以支持重复问题检索"],
"qq": ["表征这个问题以支持重复问题检索"],
"xzfaq-quality": ["表征这个商品质量问题以支持同类问题检索"],
"qr": ["表征这个电商领域的用户问题以支持相关回复检索", "表征这个电商领域的回复以支持问答检索"],
"gcls": ["表征这个电商领域的用户对话内容以支持同类对话内容检索"],
"senti": ["表征这个情绪问题以支持同类情绪问题检索"],
"scene": ["表征这个场景问题以支持同类场景问题检索"],
"hy-bx": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-dn": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-hfp": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-kt": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-sjpj": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-yzrsq": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-xyj": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-cfxd": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-dspb": ["表征这个电商领域的问题以支持同类问题检索"],
"hy-nfyyp": ["表征这个电商领域的问题以支持同类问题检索"],

# package IFT datasets
#BELLE
"BelleGroup/train_3.5M_CN": ["表征这个对话上下文以支持对话检索", "表征这个对话回复以支持检索"],
"BelleGroup/generated_chat_0.4M": ["表征这个生成对话的指令需求以支持对话检索", "表征对话内容以支持对话需求检索"],
"BelleGroup/school_math_0.25M": ["表征这个数学问题以支持答案检索", "表征这个数学解题答案以支持检索"],
"BelleGroup/train_2M_CN": ["表征这个指令需求以支持回复检索", "表征这个问答回复以支持检索"],
"BelleGroup/train_1M_CN": ["表征这个指令需求以支持回复检索", "表征这个问答回复以支持检索"],
"BelleGroup/train_0.5M_CN": ["表征这个指令需求以支持回复检索", "表征这个问答回复以支持检索"],
"BelleGroup/multiturn_chat_0.8M": ["表征这个人机多轮对话内容以支持答案检索", "表征这个答案以支持对话回复检索"],

# GPT4-based self-instruct dataset
"shibing624/alpaca-zh": ["表征这个指令需求以支持回复检索", "表征这个问答回复以支持检索"],

# 2-Turbo API based multi-round dialogue mimic(human) data
"stingning/ultrachat": ["表征这个多轮对话内容以支持答案检索", "表征这个答案以支持对话回复检索"],

# fnlp moss sft dataset
"fnlp/moss-003-sft-no-tools": ["表征这个人机多轮对话内容以支持答案检索", "表征这个答案以支持对话回复检索"],

# Firely dataset
"YeungNLP/firefly-train-1.1M": ["表征这个指令需求以支持回复检索", "表征这个问答回复以支持检索"],

}


