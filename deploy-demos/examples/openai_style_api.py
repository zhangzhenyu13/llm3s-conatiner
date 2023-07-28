def test_openai_service():
    import openai
    openai.api_base = "http://server-ip:port/v1"
    openai.api_key = "none"
    modelId="your-org/bloomS2.1" # "aquilachat-7b" #"vicuna/7b" #"baichuan-inc/baichuan-7B"  #"your-org/bloomS2.1-rlhf-v1" # 
    query = "你好！用关键词：爱情、珍贵、永恒、价值，为周大福写个100字广告。"

    def streaming_test():
        for chunk in openai.ChatCompletion.create(
            temperature=0.95, top_p= 0.7, decoder='sample',
            max_length = None, repetition_penalty=1.0,
            model= modelId,
            messages=[
                {"role": "user", "content": query}
            ],
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    def request_api():
        response = openai.ChatCompletion.create(
            temperature=0.95, top_p= 0.7, decoder='sample',
            max_length = 128, repetition_penalty=1.0,
            model=modelId,
            messages=[
                {"role": "user", "content": query}
            ],
            stream=False
        )
        print(type(response))
        if hasattr(response.choices[0].message, "content"):
            print(response.choices[0].message.content)
    
    def test_embeddings():
        texts = [
            "你好", "介绍下蓝牙5/0"
        ]
        model = "moka-ai/m3e-base"
        response =  openai.Embedding.create(input = texts, model=model,max_length = 128)
        print("res:", response.keys(), )
        for x in response['data']:
            print(x.keys())
        print(len(response['data']))
        embs = [x['embedding'] for x in response['data'] ]
        [print(emb[:10]) for emb in embs]

    test_embeddings()#;exit(0)
    request_api()
    streaming_test()
if __name__ == '__main__':
    # test_legacy_service()
    test_openai_service()
    # ...
