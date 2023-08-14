# export CUDA_VISIBLE_DEVICES=2,3
NCCL_DEBUG=INFO 

# torchrun --nproc-per-node 2 --nnodes 2 \
# python    finetune.py 
# exit 0

cd $(dirname $0)
workPath=$(pwd)
echo runPath=$workPath

configPath=$workPath/configs
echo $workPath
echo $configPath
echo "now running python"
deepspeed \
    --hostfile=$configPath/hostfile \
    --include="11.70.129.225:0,3" \
    $workPath/iFT-contrastive.py \
    --work_space $HOME \
    --model_config_file $configPath/Bert_config.json  \
    --deepspeed $configPath/deepspeed_config.json 

