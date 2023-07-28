export OPENAI_API_KEY=$YOUR_KEY

# if you have proxy running in your server, you can uncomment below to boost speed
# export http_proxy="socks5h://127.0.0.1:1080"
# export https_proxy="socks5h://127.0.0.1:1080"
python chatGPT-server.py --host 0.0.0.0 --port 1238
