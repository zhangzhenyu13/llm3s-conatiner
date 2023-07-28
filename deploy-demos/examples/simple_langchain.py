import os
os.environ['OPENAI_API_BASE'] = "http://server-ip:port/v1"
os.environ['OPENAI_API_KEY'] = 'none'
import openai
from langchain.llms import OpenAI

openai.api_base = "http://server-ip:port/v1"
openai.api_key = "none"
modelId="THUDM/chatglm2-6b" 
modelId= "your-org/bloomS2.1"

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI 

#从typing库中导入必要的函数和类型声明
from typing import Any, List, Mapping, Optional

#导入所需的类和接口
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

#定义一个名为CustomLLM的子类，继承自LLM类
class CustomLLM(LLM):
    max_lengh: int = 512
    temparature: float = 0.95
    top_p: float = 0.85
    decoder: str = 'sample'

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
            model=modelId,
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

def test1():
    chat = ChatOpenAI(model_name=modelId, temperature=0.3)
    messages = [
        SystemMessage(content="你是一个专业的数据科学家"),
        HumanMessage(content="写一个Python脚本，用模拟数据训练一个神经网络")
    ]
    response=chat(messages)

    print(response.content,end='\n')


def test2():
    # WARNING completion llm is not supported, because we only use chat-sft models
    # therefore, a wrapper is given
    # 导入提示并定义PromptTemplate

    from langchain import PromptTemplate

    template = """
    您是一位专业的数据科学家，擅长构建深度学习模型。
    用几行话解释{concept}的概念
    """
    # llm = OpenAI(model_name=modelId)
    llm = CustomLLM()
    prompt = PromptTemplate(
        input_variables=["concept"],
        template=template,
    )

    # 用PromptTemplate运行LLM
    # res =llm(prompt.format(concept="autoencoder"))
    # print(res)
    # res2= llm(prompt.format(concept="regularization"))
    # print(res2)

    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # 只指定输入变量来运行链。
    print(chain.run("autoencoder"))

    second_prompt = PromptTemplate(
        input_variables=["ml_concept"],
        template="把{ml_concept}的概念描述转换成用500字向我解释，就像我是一个五岁的孩子一样",
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    # 用上面的两个链定义一个顺序链：第二个链把第一个链的输出作为输入

    from langchain.chains import SimpleSequentialChain
    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

    # 只指定第一个链的输入变量来运行链。
    explanation = overall_chain.run("autoencoder")
    print(explanation)



if __name__ == "__main__":
    # test1()
    test2()

