model_name_or_path=$HOME/CommonModels/your-org/bloomS2
data_path=$HOME/SharedData/Bloom7Bz/data_dir/

export CUDA_VISIBLE_DEVICES="0"
python inference-model.py \
    --dev_file $data_path/test.json \
    --model_name_or_path $model_name_or_path \
    --output_file data_dir_test/predictions.json


