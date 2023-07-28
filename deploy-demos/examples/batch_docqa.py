import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, OpenAIChat
from langchain.callbacks import get_openai_callback
from langchain.embeddings.base import Embeddings
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from typing import Mapping, Optional, List, Any
import argparse
import json
import tqdm

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
class ModelHolder:
    embeddings: OpenAIEmbeddings
    llm: OpenAI

class OurEmbeddings(Embeddings):
    def __init__(self, max_length=256, model=None) -> None:
        self.max_length = max_length
        self.model = model
    def embed_query(self, text: str):
        response =  openai.Embedding.create(
            input = [text], model=self.model,
            max_length = self.max_length
        )
        embs = [x['embedding'] for x in response['data'] ]
        return embs[0]
    def embed_documents(self, texts):
        response =  openai.Embedding.create(
            input = texts, model=self.model,
            max_length = self.max_length
        )
        embs = [x['embedding'] for x in response['data'] ]
        return embs

#定义一个名为CustomLLM的子类，继承自LLM类
class CustomLLM(LLM):
    max_lengh: int = 512
    temparature: float = 0.95
    top_p: float = 0.85
    decoder: str = 'sample'
    model: str = None

    # 用于指定该子类对象的类型
    @property
    def _llm_type(self) -> str:
        return "custom"

    # 重写基类方法，根据用户输入的prompt来响应用户，返回字符串
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = openai.ChatCompletion.create(
            temperature=self.temparature, top_p= self.top_p, decoder=self.decoder,
            max_length = self.max_lengh, repetition_penalty=1.0,
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        # print(type(response))
        if hasattr(response.choices[0].message, "content"):
            # print(response.choices[0].message.content)
            return response.choices[0].message.content


    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
      
def set_env():
    import openai
    import os
    os.environ['OPENAI_API_KEY'] = 'none'
    openai.api_base = "http://11.70.129.225:7690/v1"
    # openai.api_base = "http://11.70.129.225:6200/v1"

    openai.api_key = "none"
    
    # modelId="THUDM/chatglm2-6b" 
    modelId= "your-org/bloomS2.1-FT"

    # llm = OpenAIChat(model_name = modelId)
    llm = CustomLLM(model= modelId, max_lengh=2048)
    ModelHolder.llm = llm

    embedder = "moka-ai/m3e-base"
    embeddings = OurEmbeddings(max_length=512, model=embedder)
    ModelHolder.embeddings = embeddings


set_env()



def run_test(queries:list, document:str=""):
    results = []
    text_splitter = CharacterTextSplitter(
        separator= "。", #"\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    if isinstance(document, str):
        chunks = text_splitter.split_text(document)
    else:
        input("using naive chunks")
        chunks = document
    print("chunk-size:", len(chunks))
    print("chunks:",[len(_) for _ in chunks])
    [print(i, "####\t",chunks[i]) for i in range(len(chunks))]

    # create embeddings
    embeddings = ModelHolder.embeddings
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    for query in queries:
        docs = knowledge_base.similarity_search(query)
        llm = ModelHolder.llm
        chain = load_qa_chain(llm, chain_type="stuff")
        try:
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
        except Exception as e:
                response = f"ERROR: {e.args}"
        yield response    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--doc", type=str)
    parser.add_argument("--no-splitter", action="store_true", default=False)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    lines = []
    if args.doc.endswith(".pdf"):
        pdf_reader = PdfReader(args.doc)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        with open(args.doc) as f:
            if args.no_splitter:
                for line in f:
                    if line.strip():
                        lines.append(line.strip() )
            else:
                text = f.read()

    questions = []
    with open(args.input) as f:
        for line in f:
            if line and line.strip():
                questions.append(line.strip())
    
    answers = run_test(queries=questions,  document=lines if args.no_splitter else text)

    progress_bar = tqdm.tqdm(range(len(questions)))
    with open(args.output, 'w') as fw:
        for i, a in enumerate(answers):
            q = questions[i]
            fw.write(json.dumps({"q": q, "a": a}, ensure_ascii=False)+"\n")
            progress_bar.update(1)