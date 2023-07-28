export CUDA_VISIBLE_DEVICES=2
python service_bloom.py --service-model-id "your-org/bloomS2.1-rlhf-v1"  
# python service_baichuan.py --service-model-id "baichuan-inc/baichuan-7B" 
# python service_vicuna.py --service-model-id "vicuna/7b" 

# python service_reward.py --service-model-id "your-org/rewardS1"

# python service_aquila.py --service-model-id aquilachat-7b

# python service_chatglm.py --service-model-id "THUDM/chatglm2-6b"

export CUDA_VISIBLE_DEVICES=1
python service_embedding_m3e.py --service-model-id "moka-ai/m3e-base" 

