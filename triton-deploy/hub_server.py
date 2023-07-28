from fastapi import (
    FastAPI, 
    File, UploadFile, Form,
    Request, HTTPException
)
from starlette.responses import FileResponse, Response
from pydantic import BaseModel, Field
import json
import os

with open("configs/model-hub.json") as f:
    server_config: dict = json.load(f)
model_hub = os.path.join(os.environ['HOME'], *server_config['hub_path'])
os.makedirs(model_hub,exist_ok=True)


app = FastAPI(name="model-hub")


@app.post(f"/{server_config['upload']}/")
async def upload_file(file:UploadFile = File(...), model_id: str = Form(...) ):
    # model_id = upfile.filename # requests: (filename, file)
    # model_id = file.headers.get("model_id") # wrong

    print(file.headers)
    print(file.filename, model_id)
    model_path = os.path.join(model_hub, model_id+".tgz")
    write_size = 0
    with open(model_path, "wb") as fw:
        while True:
            chunk = file.file.read(1024)
            if not chunk:break
            write_size += fw.write(chunk)
            print("read bytes:", write_size)
    return {"file-size": f"{write_size/(2**20) } MB"}

@app.post(f"/{server_config['download']}/")
async def download_file(model_id: str = Form(...)):
    print("download model-id:", model_id)
    model_path = os.path.join(model_hub, model_id+".tgz")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError("not found")
        return FileResponse(model_path, filename=os.path.basename(model_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=server_config['port'], workers=1)
