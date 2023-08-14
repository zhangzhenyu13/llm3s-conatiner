from mteb import MTEB
from typing import List, Protocol
from mteb import MTEB, AbsTask

from mteb_xz.tasks import (
    XZFaqQuality, 
    XZHyClassifier,
    XZClassifier,
    XZQQPairs
)

from mteb_zh.tasks import (
    GubaEastmony,
    IFlyTek,
    JDIphone,
    MedQQPairs,
    StockComSentiment,
    T2RReranking,
    T2RRetrieval,
    TNews,
    TYQSentiment,
    TaskType
)

from configs.instructions import instructionsMapper
from simcse_model import SimCSE

class MyModel(Protocol):
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass

clsmethod= "kNN" #"kNN"  #"logReg"

open_s2s_tasks: List[AbsTask] = [
    TYQSentiment(method=clsmethod),
    TNews(method= clsmethod),
    JDIphone(method= clsmethod),
    StockComSentiment(method= clsmethod),
    GubaEastmony(method= clsmethod),
    IFlyTek(method= clsmethod),
    
    MedQQPairs(),
]
open_tranking: List[AbsTask] =[
    # T2RReranking(2),
    T2RRetrieval(10000),
    # T2RRetrieval(100000),
    # T2RRetrieval(500000),
]

xz_s2s_tasks: List[AbsTask] = [
    XZFaqQuality(method=clsmethod),
    XZHyClassifier(taskname= "hy-bx", method=clsmethod),
    XZHyClassifier(taskname= "hy-dn", method=clsmethod),
    XZHyClassifier(taskname= "hy-hfp", method=clsmethod),
    XZHyClassifier(taskname= "hy-kt", method=clsmethod),
    XZHyClassifier(taskname= "scene", method=clsmethod),
    XZHyClassifier(taskname= "senti", method=clsmethod),
    XZClassifier(taskname = "gcls",  method=clsmethod),
    XZQQPairs(taskname = "kfqq"),
    XZQQPairs(taskname = "qq"),
    XZQQPairs(taskname = "qr"),
    XZQQPairs(taskname = "xzqq")
]

task_defualt_mapper={
    TaskType.Reranking :"qd-retriever",
    TaskType.Retrieval: "qd-retriever",
    TaskType.Classification: "cls-retriever",
    TaskType.PairClassification: "qq-retriever"
}
def inject_instructions(tasks:List[XZClassifier]):
    for task in tasks:
        desc= task.description
        task_type = desc['type']
        task_name = desc['name']
        if task_name in instructionsMapper:
            insts= instructionsMapper[task_name]
        else:
            insts = instructionsMapper[task_defualt_mapper[task_type] ]
            print("warning:use standard", )
        
        task.set_instructions(insts, task_type)
        print("added insts:", [task_name, task_type, insts])


def inject_retriver(tasks:List[XZClassifier]):
    insts = instructionsMapper['flagretriver']
    for task in tasks:
        
        desc= task.description
        task_type = desc['type']
        task_name = desc['name']
        
        task.set_instructions(insts, task_type)
        print("added insts:", [task_name, task_type, insts])


def evaluate_xz():
    # tasks = xz_s2s_tasks + open_s2s_tasks + open_tranking
    # tasks= open_s2s_tasks
    # tasks = xz_s2s_tasks

    # tasks = [XZQQPairs(taskname = "qr"),]
    tasks = open_tranking
    inject_retriver(tasks)

    
    if args.add_instruction:
        inject_instructions(tasks)
    

    evaluation = MTEB(tasks=tasks, num_max_passages=10, )
    
    evaluation.run(model, output_folder=save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pooler", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--add-instruction", action="store_true")#type=str, required=True)

    args = parser.parse_args()
    # if args.add_instruction == "Y":
    #     args.add_instruction= True
    # else:
    #     args.add_instruction= False

    print(args)
    model =  SimCSE(args.model_path, pooler=args.pooler,)
    model.half()
    save_path = f"results/result-{args.run_name}"
    
    evaluate_xz()

