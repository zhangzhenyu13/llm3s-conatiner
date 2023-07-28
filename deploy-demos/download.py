import os



def download_hf_models():
    os.environ['http_proxy'] = 'socks5h://127.0.0.1:1080'
    os.environ['https_proxy'] = 'socks5h://127.0.0.1:1080'

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    

    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # model_id = "fnlp/moss-moon-003-sft"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).half()

    # model_id="lmsys/vicuna-7b-delta-v1.1"
    # model_id = "tiiuae/falcon-7b-instruct"
    # model_id = "decapoda-research/llama-7b-hf" #"yahma/llama-7b-hf"
    # model_id = "baichuan-inc/baichuan-7B"
    model_id = "THUDM/chatglm2-6b"
    model_id = "openchat/openchat_v2_w"

    if "llama" in model_id.lower():
        from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(model_id, trust_remote_code=True).half()
    elif "chatglm" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).half()

    print(model)
    print(tokenizer)

    savePath = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    tokenizer.save_pretrained(savePath)
    model.save_pretrained(savePath)


def download_aquila():

    from baai_modelhub import AutoPull

    model_id = 'AquilaChat-7B'
    savePath = os.path.join(os.environ['HOME'], 'CommonModels', model_id)
    auto_pull = AutoPull()

    auto_pull.get_model(model_name= model_id, model_save_path=savePath )

download_hf_models()

# download_aquila()
