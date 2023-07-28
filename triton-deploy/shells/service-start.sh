model_id=$1
model_id="your-org/$model_id-FT"
echo "service for $model_id "
mkdir -p logs

if [ $model_id = your-org/chatrecV1-FT ]; then
echo "model: chatrecV1"
nohup python3 triton-proxy-sever.py \
    --service-model-id your-org/chatrecV1-FT \
    --service-port 6200 > logs/chatrec.log &
elif [ $model_id = your-org/bloomS2.1-FT ]; then
echo "model: bloomS2.1"
nohup python3 triton-proxy-sever.py \
    --service-model-id your-org/bloomS2.1-FT \
    --service-port 6200 > logs/bloom.log &
else
echo "service model-id not supported"
fi
echo "finished starting api-service"
