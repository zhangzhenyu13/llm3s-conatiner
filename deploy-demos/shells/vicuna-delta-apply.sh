modelPath=$HOME/CommonModels/
llama_id=decapoda-research/llama-13b-hf
vicuna_id=lmsys/vicuna-13b-delta-v1.1/
python3 -m fastchat.model.apply_delta \
    --base-model-path $modelPath/$llama_id \
    --target-model-path $modelPath/vicuna/13b \
    --delta-path $modelPath/$vicuna_id