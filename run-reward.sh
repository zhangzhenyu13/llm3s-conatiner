NCCL_DEBUG=INFO 

workPath=$(pwd)
configPath=$workPath/run_config
echo $workPath
echo $configPath
echo "now running python"
deepspeed \
    --hostfile=$configPath/hostfile \
    --include="11.70.129.225:0,2" \
    $workPath/reward-post-bloom.py \
    --work_space $HOME \
    --model_config_file $configPath/reward-Bloom_config.json  \
    --deepspeed $configPath/deepspeed_config.json

    # --lora_hyperparams_file=$configPath/lora_hyperparams_bloom.json \
    # --use_lora 
