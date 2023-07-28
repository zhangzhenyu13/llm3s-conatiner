import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_VISIBLE_DEVICES']='1'
model_id="vicuna/7b"
modelPath = os.path.join(os.environ['HOME'], 'CommonModels', model_id)

from fastchat.model.model_adapter import load_model


model, tokenizer = load_model(model_path=modelPath, device="cuda", 
    num_gpus=1, load_8bit=True, cpu_offloading=True)

print(tokenizer)
print(model)

