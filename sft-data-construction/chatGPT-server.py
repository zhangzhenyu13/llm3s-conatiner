from fastapi import FastAPI, Request
import uvicorn
import json
import service_utils

top_p = 1.0
temperature=1.0
decoding_args = service_utils.OpenAIDecodingArguments(
    temperature=temperature,
    n=1,
    max_tokens=1024,  # hard-code to maximize the length. the requests will be automatically adjusted
    top_p=top_p,
    stop=["\n20", "20.", "20."],
)

app = FastAPI()

# openai.api_key = os.getenv("OPENAI_API_KEY")


@app.post("/ouryx05private")
async def index(request: Request):
    requestData = await request.json()
    requestData = json.loads(json.dumps(requestData))
    prompts = requestData['prompts']
    api = requestData['api']
    model_name = requestData['model_name']
    batch_size = requestData['batch_size']

    decoding_args.max_tokens = requestData['max_tokens']
    decoding_args.top_p = requestData.get("top_p", top_p)
    decoding_args.temperature = requestData.get("temperature", temperature)

    response_texts = service_utils.openai_completion(
                prompts=prompts,
                api=api,
                model_name=model_name,
                batch_size=batch_size,
                decoding_args=decoding_args,
                logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
    
    answer = {"generations": response_texts }
    
    return answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)



