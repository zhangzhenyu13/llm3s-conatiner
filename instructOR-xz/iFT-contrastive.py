import os
import sys
import argparse, logging

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset, DatasetDict
import transformers
from transformers.trainer_utils import is_main_process
import json

from transformers import AutoTokenizer
from embedder import (
    EmbedderForPairInBatchNegTrain, 
    EmbedderForTripletInBatchNegTrain,
    ContrastiveEmbedder,
    MultiTaskTrainer
)
from dataloader import (
    RecordType, PairRecord, TripletRecord, ScoredPairRecord,
    PairCollator, 
    TripletCollator, 
    ScoredPairCollator,
    ConstrastiveCollator
)
from dataloader import (
    load_dataset_local,
    MultiTaskDataset,
    get_chr_r
)

def get_logger(logger_name,output_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) 
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir,'log.txt'),mode='w')
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(console_handler)
    return logger


def train():
    def check_encoders(modelinfo):
        encoders= ["bert", "roberta"]
        for encname in encoders:
            if encname in modelinfo:
                return True
        return False
    

    model_config = json.load(open(args.model_config_file))
    model_type = model_config['model_type']
    model_name_or_path = os.path.join(args.work_space, model_config['model_name_or_path']) 
    data_path = os.path.join(args.work_space,  model_config['data_path'])
    output_dir = os.path.join(args.work_space,  model_config['output_dir'] )
    cutoff_len = model_config['cutoff_len']
    dataset_type: str = model_config['dataset_type']  # if False, masks out inputs in loss
    pooling_strategy:str = model_config['pooling_strategy']
    temperature: float = model_config['temperature']
    group_by_length: bool = model_config['group_by_length']  # faster, but produces an odd training loss curve,
    # resume_from_checkpoint: str = None  # either training checkpoint or final adapter
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = model_config['batch_size'] // model_config['per_device_train_batch_size'] if "gradient_accumulation_steps" not in model_config else model_config['gradient_accumulation_steps']

    device_map = "auto"
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)


    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=model_config['per_device_train_batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=model_config['warmup_ratio'],
            num_train_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            logging_steps=model_config['logging_steps'],
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=model_config["save_steps"],
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=False,
            fp16 = model_config['fp16'],
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=args.deepspeed if not args.use_lora else None,
            group_by_length=group_by_length,
            report_to='none',
            remove_unused_columns =False
        )
    
    # meta info
    logger = get_logger("train", output_dir)
    logger.info("args.__dict__ : {}".format(args.__dict__))
    print("world Size----->", [training_args.world_size, world_size])

    for key, value in model_config.items():
        logger.info("{} : {}".format(key, value))
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    load_in_8bit = True if args.use_lora else False
    
    
    if check_encoders(model_type.lower()) :
        # print("test---->", device_map)
        device_map= None
        model = ContrastiveEmbedder.from_pretrained(
            model_name_or_path,
            load_in_8bit = load_in_8bit,
            device_map=device_map,
            pooling_strategy= pooling_strategy
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"{model_type}: not supported")
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "left" 
    model.config.pad_token_id = tokenizer.pad_token_id

    data_files = {
        "train": data_path #os.path.join(data_path, "train.json"),
        # "dev": os.path.join(data_path, "dev.json")
    }
    data_files = {get_chr_r(i): os.path.join(data_path, fn) 
            for i, fn in enumerate(os.listdir(data_path) ) 
    }
    
    try:
        raw_data = load_dataset("json", data_files=data_files)
    except:
        raw_data = load_dataset_local(data_path)
    raw_data = DatasetDict(
        train=MultiTaskDataset(
        raw_data.values(),
        rank=dist.get_rank(),
        world_size=training_args.world_size
        )
    )
    print(raw_data)
    [print(raw_data['train'][i]) for i in range(3) ]
    

    data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    if dataset_type == RecordType.PAIR:
        model = EmbedderForPairInBatchNegTrain(embedder=model, temperature=temperature)
        data_collator = PairCollator(tokenizer, pad_to_multiple_of=8, 
                    return_tensors="pt", padding=True, max_length=cutoff_len)
    elif dataset_type == RecordType.TRIPLET:
        model = EmbedderForTripletInBatchNegTrain(embedder=model, temperature= temperature)
        data_collator = TripletCollator(tokenizer, pad_to_multiple_of=8, 
                    return_tensors="pt", padding=True, max_length=cutoff_len)
    else:
        raise ValueError(f"datatype not supported:{dataset_type}" )
    
    
    def process_dataset(datapoint):
        if dataset_type == RecordType.PAIR:
            return PairRecord(**datapoint)
        
    # with training_args.main_process_first(desc="dataset prompt tokenize&prompt"):
    #     processed_dataset = raw_data.map(
    #         process_dataset,
    #         # batched=True, batch_size=1000,
    #         num_proc=32
    #     )
    processed_dataset = raw_data
    
    train_data = processed_dataset['train']
    val_data = None #processed_dataset['dev']
    

    print("start train...", len(train_data))
    # trainer = transformers.Trainer(
    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        print("compiled torch2 model")
    print("trainer.train")
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
    

    if is_main_process(int(os.environ.get("LOCAL_RANK", args.local_rank)) ):
        logger.info("Save checkpointing...")
        logger.info("local-rank: {}".format(int(os.environ.get("LOCAL_RANK", args.local_rank))) )
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above when using lora to train, please disregard :)")
    logger.info("Training succeeded")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--lora_hyperparams_file", default="", type=str, help="Provide it when use_lora=True")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use lora")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--work_space", type=str, required=True)
    args = parser.parse_args()
    print(f"RANK-INFO:{os.environ['LOCAL_RANK']}(/{args.local_rank})/{os.environ['RANK']}")

    train()
