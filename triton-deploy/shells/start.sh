model_id=$(head -1 configs/model.task)
export NCCL_LAUNCH_MODE=PARALLEL

cd /export/App/triton-server/
bash shells/process-model.sh
echo "start the server...for $model_id"
mkdir -p /export/Logs/

# sleep 1000 # for debugging
bash shells/service-start.sh $model_id


CUDA_VISIBLE_DEVICES=0 /opt/tritonserver/bin/tritonserver \
    --model-repository=/export/triton-model-store/model \
    --backend-config=python,shm-region-prefix-name=prefix1_ \
    --grpc-port 8500 --http-port 8501 --metrics-port 12345 \
    --log-verbose 1 --log-file /export/Logs/triton_server_gpu0.log 
