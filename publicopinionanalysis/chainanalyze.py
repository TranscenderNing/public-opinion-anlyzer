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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 翻译 英译中
class TranslateEn2Zh():
    def __init__(self):
        #self.translation_pipeline = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
        self.translation_pipeline = pipeline('translation_en_to_zh', model='/home/ldn/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-zh/snapshots/a4193836671069f1f80ce341f9227a850ffb52d4')
        
    def process_data(self,input_text):
        response = self.translation_pipeline(input_text)
        translated_text = response[0]['translation_text']
        return translated_text

# 翻译类 中译英
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

# 分析的chain
def analysis_chain(llm,prompt_template,content):
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    ans = llm_chain.predict(content = content)
    return ans
    
# 获取各个任务的prompt
def getPrompts():
    prompts = []
    sentiment_prompt = '''
        {content}
        Help me analyze the sentiment of the above public opinion is positive, negative, or neutral. You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    province_prompt = '''
        {content}
        Which province does the above public opinion belong to? Please note that you can only select one of the following provinces: Beijing, Tianjin, Hebei, Shanxi, Inner Mongolia, Liaoning, Jilin, Heilongjiang, Shanghai, Jiangsu, Zhejiang, Anhui, Fujian, Jiangxi, Shandong, Henan, Hubei, Hunan, Guangdong, Guangxi, Hainan, Chongqing, Sichuan, Guizhou, Yunnan, Tibet, Shaanxi, Gansu, Qinghai, Ningxia, Xinjiang, Taiwan, Hong Kong, Macau and uncertain. You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    keyword_prompt = '''
        {content}
        What are the key words in the above public opinion? You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    business_prompt = '''
        {content}
        What are the business details involved in the above public opinion? You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    impact_prompt = '''
        {content}
        What are the impacts of the above public opinion on users? You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    major_prompt = '''
        {content}
        Does the above public opinion constitute a major public opinion? You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    resolution_prompt = '''
        {content}
        Does the above public opinion require manual follow-up and resolution? You should provide your analysis process and answer separately.
        The analysis process:
        The answer:
    '''
    prompts.extend([sentiment_prompt,province_prompt,keyword_prompt,business_prompt,impact_prompt,major_prompt,resolution_prompt])
    return prompts


# 根据内容生成分析结果并写入文件
def getResults(contents,translatorZh2en,translator,llm,prompts):
    results = []
    n = 3
    numRelated = 0
    for i in range(0, n):
        time_start = time.time()
        print(f'----------------------process {i+1} / {n} ------------------------------\n')
        result = {}
        result['content'] = contents[i]
        isRelated = judge_relation(contents[i])
        if 'yes' in isRelated.lower():
            numRelated += 1
            result['isRelated'] = '相关'
            tasks = ['sentiment', 'province', 'keywords', 'businessDetails', 'impactOnUsers', 'isMajor', 'manualFollowUp']
            analysis_dict = {}
            content = translatorZh2en.process_data(contents[i])
            for index, task in enumerate(tasks):
                print(f'**************************处理{task}任务**************************\n')
                ans_inenglish = analysis_chain(llm=llm,prompt_template=prompts[index],content=content)
                print(f"{ans_inenglish}\n")
                analysis_dict[task] = translator.process_data(ans_inenglish)
            result['analysis'] = analysis_dict
        else: 
            result['isRelated'] = '不相关'
            result['analysis'] = ''
        results.append(result)
        print(result)
        time_end = time.time()
        print(f'-------------------------process end  耗时{(time_end - time_start)/60}分--------------------------------\n')    



    # 4. 将结果写到文件中
    with open("resultschain.txt", "w") as file:
        file.write(json.dumps(results, ensure_ascii=False, indent=4))
        

def start():
    start_time = time.time()
    # 翻译工具 英译中
    translator = TranslateEn2Zh()
    # 中译英
    translatorZh2en = TranslateZh2En()
    # 初始化模型
    llm = Vicuna7b.from_llm()
    # llm = OpenAI(openai_api_key = "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",temperature=0)
    # 获取舆情内容
    contents = get_data()
    # 获取各项的prompt
    prompts = getPrompts()
    getResults(contents,translatorZh2en,translator,llm,prompts)
    end_time = time.time()
    print(f'执行该程序耗费的时间是{(end_time-start_time)/60}分')
    
        
start()
    





