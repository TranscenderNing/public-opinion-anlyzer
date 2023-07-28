from langchain.tools import BaseTool
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,AutoModelForSeq2SeqLM
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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 翻译
class TranslateEn2Zh():
    def __init__(self):
        #self.translation_pipeline = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
        self.translation_pipeline = pipeline('translation_en_to_zh', model='/home/ldn/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-zh/snapshots/a4193836671069f1f80ce341f9227a850ffb52d4')
        
    def process_data(self,input_text):
        response = self.translation_pipeline(input_text)
        translated_text = response[0]['translation_text']
        return translated_text

# 翻译工具
translator = TranslateEn2Zh()



class TranslateZh2En():
    def __init__(self):
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("/home/ldn/.cache/huggingface/hub/models--facebook--wmt21-dense-24-wide-x-en/snapshots/9aec49e2214a8cb0bd2bf62a0b91178ca9ae2929")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/ldn/.cache/huggingface/hub/models--facebook--wmt21-dense-24-wide-x-en/snapshots/9aec49e2214a8cb0bd2bf62a0b91178ca9ae2929")
        self.tokenizer.src_lang = "zh"
        
    def process_data(self,input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        generated_tokens = self.model.generate(**inputs)
        res = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return res

# 中文-》英文翻译工具
translatorZh2en = TranslateZh2En()



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



# 解析大模型的输出
def parseLlmOutput(text,translate_chain):
    # text = text.replace("\n", " ")
    matches = ['1.','2.','3.','4.','5.','6.','7.']
    matchIdxes = []
    for index,match in enumerate(matches):
        matchIdxes.append(text.find(match))

    result = {}
    tasks = ['sentiment', 'province', 'keywords', 'businessDetails', 'impactOnUsers', 'isMajor', 'manualFollowUp']
    for i in range(len(matches)):
        if i == len(matches) - 1:
            content = text[matchIdxes[i]+2:len(text)]
        else:
            content = text[matchIdxes[i]+2:matchIdxes[i+1]]
        idx = content.find('\nReasoning process:')
        if idx != -1:
            content1 = content[0:idx]
            content2 = content[idx+19:len(content)]
            print(content2)
            translateStr = translator.process_data(content1) + '    ' + "推理过程: " + translator.process_data(content2)
        else:
            translateStr = translator.process_data(content)
        result[tasks[i]] = translateStr
    return result
        
# 使用正则表达式匹配解析任务, 并按照字典返回
def toDictByRe(text, translate_chain):
    pattern = r'\d+\.\s(Conclusion:.+?)(?=\d+\. Conclusion:|\Z)'
    matches = re.findall(pattern, text,re.DOTALL)
    # 判断分析是否完整
    if len(matches) == 7:
        result = {}
        tasks = ['sentiment', 'province', 'keywords', 'businessDetails', 'impactOnUsers', 'isMajor', 'manualFollowUp']
        for i, value in enumerate(matches):
            # pattern1 = r'Conclusion:\s(.+?)\nReasoning process:\s(.+?)\n'
            pattern1 = r':\s(.+?)\nReasoning process:\s(.+?)\n'
            matches1 = re.findall(pattern1, value, re.DOTALL)
            print(value)
            print(len(matches1))
            for match1 in matches1:
                conclusion = match1[0]
                translateStr1 = translator.process_data(conclusion)
                reasoning_process = match1[1]
                translateStr2 = translator.process_data(reasoning_process)
                
            translateStr = translator.process_data(value)     
            print(f"old : {value} \n translate1: {translateStr1} \n translate1: {translateStr2}")
            result[tasks[i]] = '结论:' +  translateStr1 + ', ' + '推理过程:' + translateStr2

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


        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=1024, message='Hello! Who are you?', 
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
        # checkpoint = "lmsys/vicuna-7b-v1.3"
        checkpoint = '/home/ldn/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/ac066c83424c4a7221aa10c0ebe074b24d3bcdb6'
        args = argparse.Namespace(cpu_offloading=False, debug=False, device='cuda', gptq_act_order=False, 
          gptq_ckpt=None, gptq_groupsize=-1, gptq_wbits=16, gpus=None, load_8bit=False, 
          max_gpu_memory=None, max_new_tokens=512, message='Hello! Who are you?', 
          model_path=checkpoint, modelname='vicuna7b',
          num_gpus=1, repetition_penalty=1.0, revision='main', temperature=0)
            
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

# 初始化模型
start_time = time.time()
llm = Vicuna7b.from_llm()
# llm = OpenAI(openai_api_key = "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",temperature=0)



# translate chain: 用于将分析结果翻译成中文的chain
translate_template = "Translate the following sentence from English to Chinese. {content}"
translate_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(translate_template)
)

### 第一个prompt
# prompt_template = '''
# {content}
# Help me analyze the original content of public opinion above. For each of the following questions, please provide the reasoning process first and then present the conclusion.

# 1.
# Conclusion: Determine whether the sentiment of public opinion is positive, neutral, or negative. Please note that the answer should be limited to positive, neutral, or negative.
# Reasoning process: Analyze the tone, context, and expressions used in the content to assess the overall sentiment.

# 2.
# Conclusion: Identify the province to which the public opinion belongs.
# Reasoning process: Analyze location-specific references, demographics, or any relevant information provided in the content to determine the associated province. Please note that the selection should be limited to the following provinces: Beijing, Tianjin, Hebei, Shanxi, Inner Mongolia, Liaoning, Jilin, Heilongjiang, Shanghai, Jiangsu, Zhejiang, Anhui, Fujian, Jiangxi, Shandong, Henan, Hubei, Hunan, Guangdong, Guangxi, Hainan, Chongqing, Sichuan, Guizhou, Yunnan, Tibet, Shaanxi, Gansu, Qinghai, Ningxia, Xinjiang, Taiwan, Hong Kong, Macau, or uncertain.

# 3.
# Conclusion: Identify the key words in the public opinion. You should output the keywords.
# Reasoning process: Identify the significant terms or phrases that encapsulate the main subject, theme, or focus of the public opinion.

# 4.
# Conclusion: Identify the business details involved in the public opinion. You should output the business details or 'The public opinion does not include commercial details.'
# Reasoning process: Analyze the content to identify any relevant information regarding companies, products, services, or other aspects related to business activities.

# 5.
# Conclusion: Determine the impacts of public opinion on users. You should output the impacts or 'The public opinion does not have any impacts on users.'
# Reasoning process: Assess the potential effects, influence, or consequences that the public opinion may have on individuals, groups, or communities.

# 6.
# Conclusion: Determine whether the public opinion constitutes a major public opinion. You should output the public opinion constitutes a major public opinion or the public opinion does not constitute a major public opinion
# Reasoning process: Assess the scale, reach, significance, and importance of the public opinion to determine its classification as major or minor.

# 7.
# Conclusion: Determine whether the public opinion requires manual follow-up and resolution. You should output the public opinion requires manual follow-up and resolution or the public opinion does not require manual follow-up and resolution
# Reasoning process: Evaluate the nature, severity, and urgency of the public opinion to determine if manual intervention or resolution is necessary.
# '''




### 第二个prompt
prompt_template = '''
{content}
Help me analyze the original content of public opinion above. For each of the following questions, please provide the reasoning process first and then present the conclusion.

1.
Conclusion: Determine whether the sentiment of public opinion is positive, neutral, or negative. Please note that the answer should be limited to positive, neutral, or negative.
Reasoning process: Analyze the tone, context, and expressions used in the content to assess the overall sentiment.

2.
Conclusion: Identify the province to which the public opinion belongs.
Reasoning process: Analyze location-specific references, demographics, or any relevant information provided in the content to determine the associated province. Please note that the selection should be limited to the following provinces: Beijing, Tianjin, Hebei, Shanxi, Inner Mongolia, Liaoning, Jilin, Heilongjiang, Shanghai, Jiangsu, Zhejiang, Anhui, Fujian, Jiangxi, Shandong, Henan, Hubei, Hunan, Guangdong, Guangxi, Hainan, Chongqing, Sichuan, Guizhou, Yunnan, Tibet, Shaanxi, Gansu, Qinghai, Ningxia, Xinjiang, Taiwan, Hong Kong, Macau, or uncertain.

3.
Conclusion: Extract 3 keywords in the public opinion. You should output the keywords.
Reasoning process: Identify the significant terms or phrases that encapsulate the main subject, theme, or focus of the public opinion.

4.
Conclusion: Identify the business details involved in the public opinion. please provide your answer in one sentence.
Reasoning process: Analyze the content to identify any relevant information regarding companies, products, services, or other aspects related to business activities.

5.
Conclusion: Determine the impacts of public opinion on users. please provide your answer in one sentence.
Reasoning process: Assess the potential effects, influence, or consequences that the public opinion may have on individuals, groups, or communities.

6. 
Conclusion: Determine whether the public opinion constitutes a major public opinion. Please note that the answer must be limited to 'yes, it constitutes major public opinion' or 'no, it constitutes major public opinion'.
Reasoning process: Assess the scale, reach, significance, and importance of the public opinion to determine its classification as major or minor.

7.
Conclusion: Determine whether the public opinion requires manual follow-up and resolution. Please note that the answer must be limited to 'yes' or 'no'.
Reasoning process: Evaluate the nature, severity, and urgency of the public opinion to determine if manual intervention or resolution is necessary.
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
n = 70
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
        content = translatorZh2en.process_data(contents[i])
        result['analysis'] = llm_chain.predict(content = content) + '\n'
        results.append(result)
    else: 
        result['isRelated'] = '不相关'
        result['analysis'] = ''
    print(result)
print(f'-------------------------process end--------------------------------\n')    

# 3. 对结果进行解析并翻译
for i, result in enumerate(results):
    print(f"-----------------parse and translate {i+1} / {numRelated}-----------------------")
    # result['analysis'] = toDictByRe(result['analysis'], translate_chain)
    result['analysis'] = parseLlmOutput(result['analysis'],translate_chain)
    if not isinstance(result['analysis'], dict):
        numAnalysisFail += 1
    print(result)
print(f"-------------------parse and translate end--------------------------")

print(f'总数: {n}\n相关数/不相关数: {numRelated} / {n - numRelated}\n解析失败数/相关数: {numAnalysisFail} / {numRelated}')

# 4. 将结果写到文件中
with open("results.txt", "w") as file:
    file.write(json.dumps(results, ensure_ascii=False, indent=4))
    
end_time = time.time()
print(f'执行该程序耗费的时间是{end_time-start_time}秒')

