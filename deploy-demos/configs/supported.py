class SupportedModels:
    roled_model_ids = {
        "your-org/rewardS1": ["User", 'Bot'],
        "your-org/rewardS2": ["User", 'Bot'],
        "your-org/bloomS2": ["User", 'Bot'],
        "your-org/bloomS2.1": ["User", 'Bot'],
        "your-org/bloomS2/gptq-8bit-128g": ["User", 'Bot'],
        "your-org/bloomS2.1-7b": ["User", 'Bot'],
        "your-org/bloomS2.1-rlhf-v1": ["User", 'Bot'],
        "vicuna/7b": ["Human", "Assistant"],
        "vicuna/13b": ["Human", "Assistant"],
        "baichuan-inc/baichuan-7B": [ "人类", "助手" ],
    }
    embedder_names ={
        "moka-ai/m3e-base"
    }
    model_names= {
        "your-org/bloomS2": "model-v2",
        "your-org/bloomS2.1": "model-v3",
        "your-org/bloomS2.1-FT": "model-v3-fast",
        "your-org/bloomS2.1-rlhf-v1": "model-v3-rlhf-S1",
        "your-org/bloomS2.1-7b": "model-v3_7b",
    }
    model_ids=[
        "your-org/bloomS2",
        "your-org/bloomS2.1",
        "your-org/bloomS2.1-FT",
        "your-org/bloomS2.1-7b",
        "your-org/bloomS2.1-rlhf-v1",
        # "your-org/bloomS2/gptq-8bit-128g",
        # "tiiuae/falcon-7b-instruct",
        # "fnlp/moss-moon-003-sft",
        "baichuan-inc/baichuan-7B",
        "aquilachat-7b",
        # "THUDM/chatglm-6b",
        "THUDM/chatglm2-6b",
        "vicuna/7b",
        "vicuna/13b",
        "bigscience/bloomz-1b1",
        # "your-org/chatrecV1",
        "your-org/rewardS1",
        "your-org/rewardS2",
    ]

    model_port ={
        "THUDM/chatglm-6b": 51001,
        "THUDM/chatglm2-6b": 51002,
        # "fnlp/moss-moon-003-sft": 51101,
        "vicuna/7b": 52101,
        "vicuna/13b": 52102,
        "aquilachat-7b": 52201,
        "baichuan-inc/baichuan-7B": 53101,
        # "tiiuae/falcon-7b-instruct": 53101,
        "bigscience/bloomz-1b1": 56001,

        "your-org/bloomS2": 57006,
        "your-org/bloomS2/gptq-8bit-128g":57007,
        "your-org/bloomS2.1": 57101,
        "your-org/bloomS2.1-7b": 57102,
        "your-org/bloomS2.1-rlhf-v1": 57103,
        "your-org/bloomS2.1-FT": 6200,

        "your-org/chatrecV1": 57301,
        "your-org/rewardS1": 57601,
        "your-org/rewardS2": 57602,

        "moka-ai/m3e-base": 49001
    }

    model_host ={
        "THUDM/chatglm-6b": "11.70.128.86",
        "THUDM/chatglm2-6b": "11.167.162.21",
        "fnlp/moss-moon-003-sft": "11.70.128.86",
        "vicuna/7b": "11.70.128.85",
        "vicuna/13b": "11.70.128.85",
        "tiiuae/falcon-7b-instruct": "11.70.128.85",
        "aquilachat-7b": "11.70.128.85",
        "baichuan-inc/baichuan-7B": "11.70.128.86",
        "bigscience/bloomz-1b1": "11.167.162.21",

        "your-org/bloomS2": "11.167.162.21",
        "your-org/bloomS2.1": "11.167.162.21",
        "your-org/bloomS2.1-rlhf-v1": "11.167.162.21",
        "your-org/bloomS2/gptq-8bit-128g": "11.167.162.21",
        "your-org/bloomS2.1-7b": "11.70.128.86",
        "your-org/chatrecV1": "11.167.162.21",
        "your-org/rewardS1": "11.167.162.21",
        "your-org/rewardS2": "11.167.162.21",
        "your-org/bloomS2.1-FT": "11.70.129.225",

        "moka-ai/m3e-base": "11.70.128.86"
    }

