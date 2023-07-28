from langchain.tools import BaseTool
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from fastchat.model import load_model, get_conversation_template, add_model_args
import argparse
import torch
from langchain import LLMMathChain
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from transformers import pipeline
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import langchain
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# langchain.debug = True

## 工具使用的模型设置
class Vicuna7bForTool(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # print(f"stop is {stop}/////")
        stop = None
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")


        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=512, message='Hello! Who are you?', 
          model_path='lmsys/vicuna-7b-v1.3', modelname='vicuna7b',
          num_gpus=1, repetition_penalty=1.0, revision='main', temperature=0.7)
        # print(f"///{args.message}///")
        msg = args.message
        msg = prompt

        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs


    @classmethod
    def from_llm(
        cls
    ):
        """Initialize the BabyAGI Controller."""
        checkpoint = "lmsys/vicuna-7b-v1.3"
        
        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=512, message='Hello! Who are you?', 
          model_path='lmsys/vicuna-7b-v1.3', modelname='vicuna7b',
          num_gpus=1, repetition_penalty=1.0, revision='main', temperature=0.7)
            
        model, tokenizer = load_model(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            revision=args.revision,
            debug=args.debug,
        )
        return cls(
            model=model,
            tokenizer=tokenizer
        )


## 自定义模型 vicuna7b
class Vicuna7b(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # print(f"stop is {stop}/////")
        stop = None
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")


        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=512, message='Hello! Who are you?', 
          model_path='lmsys/vicuna-7b-v1.3', modelname='vicuna7b',
          num_gpus=1, repetition_penalty=1.0, revision='main', temperature=0.7)
        # print(f"///{args.message}///")
        msg = args.message
        msg = prompt

        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        # print(f"\n 模型输入：{msg}")
        # print(f"输入结束")
        # print(f"\n 模型输出：{outputs}")
        # print(f"输出结束\n")
        
        
        # 截断Action input
        start_index = outputs.find("Observation:")
        if start_index != -1:
            outputs = outputs[:start_index].strip()
            
            
            
        action_index = outputs.find("Action:")
        answer_index = outputs.find("Final Answer")
        # final answer不存在，而且没有输出ACTION 和 ACTION INPUT
        if answer_index == -1 and action_index == -1:
             outputs = outputs + "\n\nFinal Answer: "
        
        #  Final Answer和Action共存的情况
        if action_index != -1 and answer_index != -1:
            # 删除answer_index之前的
            if action_index < answer_index:
                outputs = outputs[answer_index:].strip()
            # 删除action_index之后的
            elif action_index > answer_index:
                outputs = outputs[:action_index].strip()
                
        
        # 替换Action Input中的内容
        action_input_index = outputs.find("Action Input:")
        if action_input_index != -1 and start_index != -1:
            outputs = outputs[:action_input_index].strip() + "\n\nAction Input:  The original content of the public opinion is '6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号'" + "\n\nObservation:"
        else:
            outputs = outputs
                

        # 替换action中的内容
        # action_start_index = outputs.find("Action:")
        # action_input_start_index = outputs.find("Action Input:")

        # if action_start_index != -1 and action_input_start_index != -1:
        #     outputs = outputs[:action_start_index].strip() + "\n\nAction: Search" + "\n\nAction Input:" + outputs[action_input_start_index+len("Action Input:"):]
        # else:
        #     outputs = outputs
        
        
        # print(f"\n 模型输入：{msg}")
        # print(f"输入结束")
        # print(f"\n 模型输出：{outputs}")
        # print(f"输出结束\n")
        
        return outputs


    
    @classmethod
    def from_llm(
        cls
    ):
        """Initialize the BabyAGI Controller."""
        checkpoint = "lmsys/vicuna-7b-v1.3"
        
        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=512, message='Hello! Who are you?', 
          model_path='lmsys/vicuna-7b-v1.3', modelname='vicuna7b',
          num_gpus=1, repetition_penalty=1.0, revision='main', temperature=0.7)
            
        model, tokenizer = load_model(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            revision=args.revision,
            debug=args.debug,
        )
        return cls(
            model=model,
            tokenizer=tokenizer
        )


llm = Vicuna7b.from_llm()
# llm = OpenAI(openai_api_key = "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",temperature=0)
# lm = OpenAI(openai_api_key="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",temperature=0)
llmTool = Vicuna7bForTool.from_llm()

## 工具1: 搜索工具
search = SerpAPIWrapper(serpapi_api_key="5c4c2c19daf6558a5d339eaeaaff161275f3f8e822bf575b9f47808f93062639")
## 工具2：初始化大语言模型的 数学工具
llm_math_chain = LLMMathChain.from_llm(
    llm=llm,
    verbose=True
)


## 工具01：舆情是否属于运营商
class Tool01(BaseTool):
    name = "JudgeTool"
    description = "useful for when you need to judge whether the public opinion pertains to a mobile network operator"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        print(query)
        response ="the public opinion dosn't  pertains to a mobile network operator."
        if "mobile network operator" in query or "移动" in query:
            response = "the public opinion  pertains to a mobile network operator."
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

## 工具02： 情感分类
class Tool02(BaseTool):
    name = "Sentiment Classification"
    description = "useful for when you need to evaluate the sentiment (positive or negative or neutral) of the public opinion"

    def load_model(self):
        classifier = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
        return classifier
    
    def process_data(self,text, classifier):
        response = classifier(text)
        return response[0]['label']

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        text = query
        # Load the model
        classifier = self.load_model()
        # Process the data
        response = self.process_data(text, classifier)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

## 工具03：舆情归属省份
class Tool03(BaseTool):
    name = "Province Determination"
    description = "useful for when you need to determine the province to which public opinion belongs"
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me determine the province to which public opinion belongs based on the following information:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   

## 工具04：舆情关键词
class Tool04(BaseTool):
    name = "Keyword Extractor"
    description = "useful for when you need to extract key words associated with the public opinion"
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me extract keywords from the following sentence:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   
## 工具05：具体业务
class Tool05(BaseTool):
    name = "Business Extractor"
    description = "useful for when you need to get business details"
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me extract specific business details from the following information:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   
## 工具06：对用户的影响
class Tool06(BaseTool):
    name = "Impact Extractor"
    description = "useful for when you need to analyze the impact on users."
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me analyze the impact on users from the following information:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   

## 工具07：是否为重大舆情
class Tool07(BaseTool):
    name = "Major Event Determination"
    description = "useful for when you need to judge if a public opinion is a major one"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me determine whether the following public opinion is a major one:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   

## 工具08：是否需要人工跟进解决
class Tool08(BaseTool):
    name = "Human Tracker"
    description = "useful for when you need to judge whether public opinion requires manual follow-up and resolution"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me determine whether the following public opinion requires manual follow-up and resolution:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   


## 工具09：输出对所有问题的总结
class Tool09(BaseTool):
    name = "SummariZer"
    description = "useful for when you summarize the results of various parts"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"Help me aggregate the results of each section based on the following content:{query}"
        response = llmTool(query)
        return response 

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
   
   
tools = [
    # Tool(
    #     name="Search",
    #     func = search.run,
    #     # description="useful for when you need to answer questions about NLP"
    #     description="useful for when you need to answer questions about current events or the current state of the world"
    # ),
    # Tool(
    #     name="Calculator",
    #     func=llm_math_chain.run,
    #     description="Useful when you need to answer questions about math."
    # ),
    Tool01(),
    Tool02(),
    Tool03(),
    Tool04(),
    Tool05(),
    Tool06(),
    Tool07(),
    Tool08(),
    Tool09(),
]

    
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
# agent.run("规划一条学习自然语言处理的路线")
# agent.run("To analyze the original content of public opinion, the following aspects need to be examined: whether the public opinion pertains to a mobile network operator, the sentiment (positive or negative) of the public opinion, the province to which the public opinion belongs, the key words associated with the public opinion, specific business details, the impact on users, whether it constitutes a major public opinion issue, and whether manual intervention is required for resolution. Finally, a summary of the analysis results is generated. The original content of the public opinion is: '6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号'")
agent.run("""6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号.
          Help me analyze the original content of public opinion above:  The following aspects need to be examined: 
          whether the public opinion pertains to a mobile network operator, 
          the sentiment (positive or negative) of the public opinion, 
          the province to which the public opinion belongs, 
          the key words associated with the public opinion, 
          the specific business details, 
          the impact on users, whether it constitutes a major public opinion issue, and whether manual intervention is required for resolution. 
          Finally, a summary of the analysis results is generated. """)