export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export http_proxy="socks5h://127.0.0.1:1080"
# export https_proxy="socks5h://127.0.0.1:1080"

# modelPath="$HOME/CommonModels/instructOR/base/"
# modelPath="$HOME/CommonModels/moka-ai/m3e-small"
modelPath="$HOME/CommonModels/BAAI/bge-base-zh"
pooler="cls"
runName="bge-base"
# runName="m3e-small"

cd $(dirname $0)
python deep_encoding.py \
    --model-path $modelPath \
    --pooler $pooler \
    --run-name $runName 

    # --add-instruction 

