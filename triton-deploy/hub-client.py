import requests
import tqdm
import argparse
import json
import os

abspath = os.path.dirname(__file__)
with open(os.path.join(abspath,"configs/model-hub.json" )) as f:
    server_config: dict = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, type=str, choices=["upload", "download"])
parser.add_argument("--model-id", required=True, type=str)
parser.add_argument("--model-file", type=str, default=None)
args = parser.parse_args()
infile= args.model_file
model_id = args.model_id

assert model_id and model_id.strip(), "null model-id"

headers = {}
# file_key = "bloom-chatrec-1b1"

def upload():
    remote_url = "http://%s:%d/%s"%(
        server_config['host'], server_config['port'], server_config['upload']
    )
    print("remote:", remote_url)
    headers['model_id'] = model_id
    with open(infile, "rb" ) as f:
        file_data = f.read()
        r = requests.post(remote_url, 
            data= {"model_id": model_id},
            files={"file": file_data} , headers=headers)
    print(r)
    print(r.text)

def download():
    remote_url = "http://%s:%d/%s"%(
        server_config['host'], server_config['port'], server_config['download']
    )
    headers = {'Content-Type': 'application/json'}
    print("remote:", remote_url)
    r = requests.post(remote_url, data={"model_id": model_id},  stream=True )
    if r.status_code == 200:
        progress_bar = tqdm.tqdm(desc="MB")
        
        with open(f"./{model_id}.tgz", "wb") as f:
            for chunk in r.iter_content(chunk_size=10*2**10):
                download_size = f.write(chunk)
                progress_bar.update(download_size/(2**20))
    else:
        print("download failed:", r)

if __name__ == "__main__":
    if args.mode == "upload":
        upload()
    elif args.mode == "download":
        download()
    else:
        raise ValueError(f"mode not supported: {args.mode}")