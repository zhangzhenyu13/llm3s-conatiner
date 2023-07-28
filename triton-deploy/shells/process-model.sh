model_id=$(head -1 configs/model.task)
python3 hub-client.py --mode download --model-id $model_id 
tar -vxf $model_id.tgz
mkdir -p /export/triton-model-store/
mv model /export/triton-model-store/model
