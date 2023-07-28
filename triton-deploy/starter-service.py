from openai_api import app
import uvicorn
import sys
import os

if __name__ == "__main__":
    with open("configs/proxy.json") as f:
        import json
        server_config= json.load(f)
    app_name = os.path.basename(sys.argv[1]).rstrip(".py")
    uvicorn.run(f"{app_name}:app", host='0.0.0.0', 
                port= server_config['service-port'], 
                workers= server_config['service-workers']
    )