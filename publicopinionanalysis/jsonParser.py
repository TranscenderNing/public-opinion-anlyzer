from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import argparse
from langchain.tools import BaseTool
import time
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

import langchain
langchain.debug = True

# model_name = "text-davinci-003"
# temperature = 0.0
# model = OpenAI(model_name=model_name, temperature=temperature,openai_api_key="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")


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


## 初始化模型
model = Vicuna7b.from_llm()



# 定义期望的数据格式
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=joke_query)

output = model(_input.to_string())

res = parser.parse(output)
print(res)

