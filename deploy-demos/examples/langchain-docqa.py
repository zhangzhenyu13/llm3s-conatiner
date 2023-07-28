import streamlit as st
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
from streamlit_chat import message
from typing import Mapping, Optional, List, Any

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
    #OpenAIEmbeddings(model= embedder, deployment="your-org")
    # print([
    #     embeddings.client, 
    #     embeddings.deployment, 
    #     embeddings.openai_api_version,
    #     embeddings.openai_api_type,
    #     embeddings.client.__dict__,
    #     embeddings._invocation_params
    #     ])
    ModelHolder.embeddings = embeddings


set_env()

if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.set_page_config(layout="wide")
col1, col2 = st.columns([1,2])

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        if prompt:
          docs = knowledge_base.similarity_search(prompt)
        llm = ModelHolder.llm
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=prompt)
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)

# Left column: Upload PDF text
col1.header("Upload PDF Text")
col1.header("Ask your PDF  ")

# upload file
pdf = col1.file_uploader("Upload your PDF", type="pdf")

# extract the text
if pdf is not None:
  pdf_reader = PdfReader(pdf)

  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
#   print([text])

  t1=f"""<font color='white'>{text}</fon>"""
  with col2:
      html(t1, height=400, scrolling=True)


  # split into chunks
  text_splitter = CharacterTextSplitter(
    separator="。", #"\n",
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
  print("chunk-size:", len(chunks))
  print("chunks:",[len(_) for _ in chunks])
  [print(i, "####\t",chunks[i]) for i in range(len(chunks))]

  # create embeddings
  embeddings = ModelHolder.embeddings
  knowledge_base = FAISS.from_texts(chunks, embeddings)

  # show user input
  st.text_input("Ask a question about your PDF:", key="user")
  st.button("Send", on_click=send_click)

   # col1.write(response)
  if st.session_state.prompts:
    for i in range(len(st.session_state.responses)-1, -1, -1):
        message(st.session_state.responses[i], key=str(i), seed='Milo')
        message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)