import openai
openai.api_base = "http://localhost:6200/v1"
openai.api_key = "none"
modelId="your-org/bloomS2.1-FT"

def request_api(query):
    response = openai.ChatCompletion.create(
        temperature=0.95, top_p= 0.7, decoder='sample',
        max_length = 128, repetition_penalty=1.0,
        model=modelId,
        messages=[
            {"role": "user", "content": query}
        ],
        stream=False, seed=2020
    )
    print(type(response))
    if hasattr(response.choices[0].message, "content"):
        print(response.choices[0].message.content)
        return response.choices[0].message.content

# request_api("hi")

if __name__ == "__main__":
    import sys, tqdm, json
    infile, outfile = sys.argv[1:3]
    with open(infile, 'r') as f, open(outfile, 'w') as fw:
        for line in tqdm.tqdm(f):
            x= json.loads(line)
            query, target = x['input'], x['output']
            response = request_api(query=query)
            fw.write(json.dumps({
                "prompt": query,
                "target": target,
                "pred": response,
            }, ensure_ascii=False)+"\n")

