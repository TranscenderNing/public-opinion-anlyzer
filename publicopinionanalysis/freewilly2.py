import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
import re
import sys

from transformers import pipeline
from typing import Any
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 翻译
class TranslateEn2Zh():
    def __init__(self):
        self.translation_pipeline = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
        
    def process_data(self,input_text):
        response = self.translation_pipeline(input_text)
        translated_text = response[0]['translation_text']
        return translated_text

# 翻译工具
translator = TranslateEn2Zh()

# 是否为英文单词
def is_english_word(word):
    return re.match("^[a-zA-Z]+$", word) is not None

# 从json文件中获取内容
def get_data():
    # 打开并读取 JSON 文件
    with open('/home/ldn/langchain/vicunaModel/publicopinionanalysis/datasets/data.json', 'r') as file:
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

# 使用正则表达式匹配解析任务, 并按照字典返回
def toDictByRe(text, translate_chain):
    pattern = r'\d+\.\s(.+?)[\r\n]+'
    matches = re.findall(pattern, text)
    # 判断分析是否完整
    if len(matches) == 7:
        result = {}
        tasks = ['sentiment', 'province', 'keywords', 'businessDetails', 'impactOnUsers', 'isMajor', 'manualFollowUp']
        for i, value in enumerate(matches):
            if i == 2:
                matchesquotes = re.findall(r'"([^"]*)"', value)
                keywords = ""
                for elem in matchesquotes:
                        keywords +=   elem + ' '
                result[tasks[i]] = keywords
                continue
            translateStr = translator.process_data(value)     
            # print(f"old : {value} \n translate: {translateStr}")
            result[tasks[i]] = translateStr
        return result 
    else:
        return "Analysis incomplete."


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

        system_prompt = "### System:\nYou are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
        message = prompt
        prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

        outputs = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return outputs

    @classmethod
    def from_llm(
        cls
    ):
        """Initialize the BabyAGI Controller."""
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/FreeWilly2", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained("stabilityai/FreeWilly2", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

        return cls(
            model=model,
            tokenizer=tokenizer
        )

# 初始化模型
start_time = time.time()
llm = Vicuna7b.from_llm()

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

# chain: 用于将分析结果翻译成中文的chain
translate_template = "Translate the following sentence from English to Chinese. {content}"
translate_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(translate_template)
)


## 1 获取舆情内容
contents = get_data()

# 2. 获取结果
results = []
n = 2
numRelated = 0
numAnalysisFail = 0
for i in range(0, n):
    result = {}
    result['content'] = contents[i]
    print(f'----------------------process {i+1} / {n}------------------------------\n')
    isRelated = judge_relation(contents[i])

    if 'yes' in isRelated.lower():
        numRelated += 1
        result['isRelated'] = '相关'
        result['analysis'] = llm_chain.predict(content = contents[i]) + '\n'
        results.append(result)
    else: 
        result['isRelated'] = '不相关'
        result['analysis'] = ''
    print(result)
print(f'-------------------------process end--------------------------------\n')    

# 3. 对结果进行解析并翻译
for i, result in enumerate(results):
    print(f"-----------------parse and translate {i+1} / {numRelated}-----------------------")
    result['analysis'] = toDictByRe(result['analysis'], translate_chain)
    if not isinstance(result['analysis'], dict):
        numAnalysisFail += 1
    print(result)
print(f"-------------------parse and translate end--------------------------")

print(f'总数: {n}\n相关数/不相关数: {numRelated} / {n - numRelated}\n解析失败数/相关数: {numAnalysisFail} / {numRelated}')

# 4. 将结果写到文件中
with open("freewilly2.txt", "w") as file:
    file.write(json.dumps(results, ensure_ascii=False, indent=4))
    
end_time = time.time()
print(f'执行该程序耗费的时间是{end_time-start_time}秒')


