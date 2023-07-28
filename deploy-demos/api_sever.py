from openai_api import app
import uvicorn

if __name__ == "__main__":
    with open("configs/server.json") as f:
        import json
        server_config= json.load(f)
    uvicorn.run("openai_api:app", host='0.0.0.0', 
                port= server_config['api'], 
                workers= server_config['api_workers']
    )