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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import langchain
# langchain.debug = True
# 从json文件中获取内容
def get_data():
    # 打开并读取 JSON 文件
    with open('data.json', 'r') as file:
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

# 初始化模型
start_time = time.time()
llm = OpenAI(openai_api_key="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",temperature=0)

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
for i in range(0,2):
    # 使用模型判断相关性
    # isRelated = judge_chain.predict(content = contents[i])
    # 使用关键词判断相关性
    isRelated = judge_relation(contents[i])
    if 'yes' in isRelated.lower():
        results[contents[i]] = parse_chain.predict(content = contents[i])
    else: results[contents[i]] = '此条舆情和中国移动不相关'


#4. 将结果写到文件中
instructions_str = 'Answer the user query.\n    The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{\"properties\": {\"sentiment\": {\"title\": \"Sentiment\", \"description\": \"answer whether the sentiment positive or negative. \", \"type\": \"string\"}, \"province\": {\"title\": \"Province\", \"description\": \"answer to which province is the sentiment associated with.\", \"type\": \"string\"}, \"keyword\": {\"title\": \"Keyword\", \"description\": \"The keywords related to the sentiment. Maybe more than one.\", \"type\": \"string\"}, \"impactOnUsers\": {\"title\": \"Impactonusers\", \"description\": \"how does the sentiment impact the user? As detailed as possible.\", \"type\": \"string\"}, \"isMajor\": {\"title\": \"Ismajor\", \"description\": \" Whether it is a major sentiment and explain why. As detailed as possible.\", \"type\": \"string\"}, \"manualFollow\": {\"title\": \"Manualfollow\", \"description\": \"Whether it require manual follow-up and explain why. As detailed as possible.\", \"type\": \"string\"}}, \"required\": [\"sentiment\", \"province\", \"keyword\", \"impactOnUsers\", \"isMajor\", \"manualFollow\"]}\n```\n    user query: Analyze the given sentiment content. provide a paragraph output of the sentiment analysis results, including the following aspects: whether the sentiment is positive or negative, the province associated with the sentiment, the keywords related to the sentiment, the impact of the sentiment on users, whether it is a major sentiment, and whether manual follow-up is required. Every aspect needs to be as detailed as possible. Sentiment content:'
instructions = []
for key, value in results.items():
    instructions.append({"instruction": instructions_str + key, "input": '', "output":value})

print('----------------------------------')
print(instructions)
print('----------------------------------')

# with open("results.txt", "w") as file:
#     # 遍历字典的键值对，将其写入文件
#     idx = 1
#     for key, value in results.items():
#         file.write(f"热点舆情内容{idx}:{key}\n分析结果:\n{value}\n\n")
#         idx += 1
    
end_time = time.time()
print(f'执行该程序耗费的时间是{end_time-start_time}秒')

