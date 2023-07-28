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
from langchain import PromptTemplate,LLMChain
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        
        
        # 截断Action input
        start_index = outputs.find("Observation:")
        if start_index != -1:
            outputs = outputs[:start_index].strip()
            
        # 替换action中的内容
        # action_start_index = outputs.find("Action:")
        # action_input_start_index = outputs.find("Action Input:")

        # if action_start_index != -1 and action_input_start_index != -1:
        #     outputs = outputs[:action_start_index].strip() + "\n\nAction: Search" + "\n\nAction Input:" + outputs[action_input_start_index+len("Action Input:"):]
        # else:
        #     outputs = outputs
        # print(f"\n 模型输出：{outputs}")
        # print(f"\n 输出结束\n")
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

## 工具1： 搜索工具
search = SerpAPIWrapper(serpapi_api_key="5c4c2c19daf6558a5d339eaeaaff161275f3f8e822bf575b9f47808f93062639")
## 工具2：初始化大语言模型的 数学工具
llm_math_chain = LLMMathChain.from_llm(
    llm=llm,
    verbose=True
)
tools = [
    Tool(
        name="Search",
        func = search.run,
        # description="useful for when you need to answer questions about NLP"
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful when you need to answer questions about math."
    ),
]


# res = llm("写一篇关于5g的报告,不少于3000字")
# res = llm("介绍一下复旦大学")
# res = llm("清华大学的英文是")
# res = llm("介绍一下北京邮电大学")
# print(res)

# 使用chain
prompt_template = "To analyze the original content of public opinion, the following aspects need to be examined: whether the public opinion pertains to a mobile network operator, the sentiment (positive or negative) of the public opinion, the province to which the public opinion belongs, the key words associated with the public opinion, specific business details, the impact on users, whether it constitutes a major public opinion issue, and whether manual intervention is required for resolution. Finally, a summary of the analysis results is generated. The original content of the public opinion is: {content}"
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)


content = "6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号"
print(llm_chain.predict(content = content))


# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
# agent.run("规划一条学习自然语言处理的路线")
