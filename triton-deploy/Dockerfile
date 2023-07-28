FROM your-org-repo/trition-server-xz:23.04-py3

# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     uvicorn fastapi openai

RUN mkdir -p /export/App/triton-server/
COPY ./ /export/App/triton-server/

ENTRYPOINT bash /export/App/triton-server/shells/start.sh
