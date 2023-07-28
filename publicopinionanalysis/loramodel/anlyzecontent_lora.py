from langchain.tools import BaseTool

import time
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
import json
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import json
import sys
# sys.path.append('/home/ldn/langchain/vicunaModel/loramodel/')
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import langchain
# langchain.debug = True

# 从json文件中获取内容
def get_data():
    # 打开并读取 JSON 文件
    with open('../datasets/data.json', 'r') as file:
        json_data = file.read()

    # 解析 JSON 数据
    parsed_data = json.loads(json_data)

    # 输出解析后的数据
    # print(parsed_data,type(parsed_data))
    return parsed_data['content']

# 判断相关性
def judge_relation(content):
    key_list = ["中国移动","移动公司","移动网络","10086"]
    for key in key_list:
        if key in content:
            return "yes"
    return "no" 

# 将1.  2.  3. 转换成json格式
def tojson(text):
    keys = ["舆情情感","舆情归属省份","舆情关键词","舆情商业细节","舆情影响","是否为重大舆情","舆情需要跟踪处理"]
    data = {}
    lines = text.split('\n')
    key_idx = 0
    for line in lines:
        if line.strip():
            parts = line.split('. ', 1)
            if len(parts) >= 2:
                key = keys[key_idx]
                key_idx += 1
                value = parts[1]
                data[key] = value

    json_data = json.dumps(data, ensure_ascii=False)
    return json_data



# 自定义模型 vicuna7b
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
        # print(f'modle:{self.model},\n{self.tokenizer}\n')
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # print(f'inputs: {inputs}\n')
        input_ids = inputs["input_ids"].to('cuda:2')
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=256, # max_length=max_new_tokens+input_sequence
            min_new_tokens=0, # min_length=min_new_tokens+input_sequence
        )
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        # print(f'output: {output}\n')
        # print(f'output length: {len(output)}')
        # output = self.tokenizer.decode(output).split("### Response:")[1].strip()
        output = self.tokenizer.decode(output)
        print(f'----------------output is {output}\n')
        return output

    @classmethod
    def from_llm(
        cls
    ):

        BASE_MODEL="/home/ldn/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/ac066c83424c4a7221aa10c0ebe074b24d3bcdb6"
        LORA_PATH="/home/public/zuoch/Chinese-Vicuna/lora-Vicuna" 
        # fix the path for local checkpoint
        lora_bin_path = os.path.join(LORA_PATH, "adapter_model.bin")
        print(lora_bin_path)
        if not os.path.exists(lora_bin_path) and args.use_local:
            pytorch_bin_path = os.path.join(LORA_PATH, "pytorch_model.bin")
            print(pytorch_bin_path)
            if os.path.exists(pytorch_bin_path):
                os.rename(pytorch_bin_path, lora_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
                )
            else:
                assert ('Checkpoint is not Found!')
                
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            # device_map="auto", 
            device_map={"": 2},
        )
        model = StreamPeftGenerationMixin.from_pretrained(
            model, LORA_PATH, torch_dtype=torch.float16, device_map={"": 2}  # device_map="auto", #device_map={"": 0}
        )

        tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
        
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
    
        return cls(
            model=model,
            tokenizer=tokenizer
        )

# 初始化模型
start_time = time.time()
llm = Vicuna7b.from_llm()
print(llm)
# llm = OpenAI(openai_api_key="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",temperature=0)

# chain1: 用于分析舆情的chain
# prompt_template = "To analyze the original content of public opinion, the following aspects need to be examined: whether the public opinion pertains to a mobile network operator, the sentiment (positive or negative) of the public opinion, the province to which the public opinion belongs, the key words associated with the public opinion, specific business details, the impact on users, whether it constitutes a major public opinion issue, and whether manual intervention is required for resolution. Finally, a summary of the analysis results is generated. The original content of the public opinion is: {content}"
# prompt_template = '''
# {content}
# Help me analyze the original content of public opinion above, You should analyze from the following aspecs and provide a corresponding response for each analysis point.
# 1.whether the public opinion pertains to a mobile network operator?
# 2.Is the sentiment of public opinion positive, neutral, or negative?
# 3.Which province does the public opinion belong to?
# 4.What are the key words in the public opinion?
# 5.What are the business details involved in the public opinion?
# 6.What are the impacts of public opinion on users?
# 7.Does the public opinion constitute a major public opinion?
# 8.Does the public opinion require manual follow-up and resolution?
# Finally, please summarize the analysis results.
# '''

prompt_template = '''
{content}
Help me analyze the original content of public opinion above, You should analyze from the following aspecs and provide a corresponding response for each analysis point.
1.Is the sentiment of public opinion positive, neutral, or negative?
2.Which province does the public opinion belong to?
3.What are the key words in the public opinion?
4.What are the business details involved in the public opinion?
5.What are the impacts of public opinion on users?
6.Does the public opinion constitute a major public opinion?
7.Does the public opinion require manual follow-up and resolution?
Finally, please summarize the analysis results.
'''
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# chain2: 用于判断是否和移动公司相关的chain
judge_template = '''Answer the following questions as best you can. 
Answer should be one of [yes, no].
Question : China Mobile Communications Group Co., Ltd., also known as China Mobile Group, is a large-scale state-owned enterprise engaged in the telecommunications industry in the People's Republic of China. China Mobile provides comprehensive communication services across all 31 provinces, autonomous regions, municipalities, and the Hong Kong Special Administrative Region in mainland China. Its primary business includes mobile voice and data services, wired broadband, as well as other communication and information services.Does the text involve China Mobile Communications Group Co., Ltd.? 
Text : {content}'''
judge_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(judge_template)
)

# chain3: 用于将分析结果翻译成中文的chain
translate_template = "Translate the following sentence from English to Chinese. {content}"
translate_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(translate_template)
)

# chain4: 用于解析成json格式的chain
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
# Define your desired data structure.
class Analyse(BaseModel):
    sentiment: str = Field(
        description="answer whether the sentiment positive or negative. "
    )
    province: str = Field(
        description="answer to which province is the sentiment associated with."
    )
    keyword: str = Field(description="The keywords related to the sentiment. Maybe more than one.")
    impactOnUsers: str = Field(
        description="how does the sentiment impact the user? As detailed as possible."
    )
    isMajor: str = Field(description=" Whether it is a major sentiment and explain why. As detailed as possible.")
    manualFollow: str = Field(description="Whether it require manual follow-up and explain why. As detailed as possible.")

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Analyse)

jsonparse_template = """
    Answer the user query.
    {format_instructions}
    user query: Analyze the given sentiment content. provide a paragraph output of the sentiment analysis results, including the following aspects: whether the sentiment is positive or negative, the province associated with the sentiment, the keywords related to the sentiment, the impact of the sentiment on users, whether it is a major sentiment, and whether manual follow-up is required. Every aspect needs to be as detailed as possible. Sentiment content: {content}
    """

parse_template = PromptTemplate(
    template=jsonparse_template,
    input_variables=["content"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

parse_chain = LLMChain(
    llm=llm,
    prompt=parse_template
)


## 1 获取舆情内容
contents = get_data()
# print(contents,type(contents),len(contents))
# content = ""
# contents.append(content)

# 2. 获取结果
results = {}
# n = len(contents)
n = 5
for i in range(0,1):
    print(f"-------------------------{i}------------------------")
    # 使用模型判断相关性
    # isRelated = judge_chain.predict(content = contents[i])
    # 使用关键词判断相关性
    isRelated = judge_relation(contents[i])
    if 'yes' in isRelated.lower():
        results[contents[i]] = parse_chain.predict(content = contents[i])
        print(f"results[contents[i]]:{results[contents[i]]}")
    else: results[contents[i]] = '此条舆情和中国移动不相关'

# 3. 对结果进行翻译并解析
for key, value in results.items():
    pass
    # 翻译
    # results[key] = translate_chain.predict(content = value)
    # print(f"翻译结果：{results[key]}\n")
    # 解析
    # results[key] = parse_chain.predict(content = results[key])
    # results[key] = tojson(results[key])

# 4. 将结果写到文件中
with open("results.txt", "w") as file:
    # 遍历字典的键值对，将其写入文件
    idx = 1
    for key, value in results.items():
        file.write(f"热点舆情内容{idx}:{key}\n分析结果:\n{value}\n\n")
        idx += 1
    
end_time = time.time()
print(f'执行该程序耗费的时间是{end_time-start_time}秒')

