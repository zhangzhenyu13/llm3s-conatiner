NCCL_DEBUG=INFO 

workPath=$(pwd)
configPath=$workPath/run_config
echo $workPath
echo $configPath
echo "now running python"
deepspeed \
    --hostfile=$configPath/hostfile \
    --include="ip-server1:0,1,2,3,4,5,6,7@ip-server2:0,1,2,3,4,5,6,7" \
    $workPath/iFT-post.py \
    --work_space $HOME \
    --model_config_file $configPath/Bloom_config.json  \
    --deepspeed $configPath/deepspeed_config.json

    # --lora_hyperparams_file=$configPath/lora_hyperparams_bloom.json \
    # --use_lora 
