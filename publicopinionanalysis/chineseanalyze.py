from langchain.tools import BaseTool
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from fastchat.model import load_model, get_conversation_template, add_model_args
import argparse
import torch
import time
from langchain import LLMMathChain
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain import PromptTemplate,LLMChain
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 判断相关性
def judge_relation(content):
    key_list = ["中国移动","移动公司","移动网络","10086"]
    for key in key_list:
        if key in content:
            return "yes"
    return "no" 


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


class CustomLLM(LLM):
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
        # # stop = None
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


    
    @classmethod
    def from_llm(
        cls,
    ):
        """Initialize the BabyAGI Controller."""
        checkpoint = "/home/ldn/langchain/model/models--THUDM--chatglm-6b/snapshots/1d240ba371910e9282298d4592532d7f0f3e9f3e"
        # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('/home/ldn/langchain/token/models--THUDM--chatglm-6b/snapshots/1d240ba371910e9282298d4592532d7f0f3e9f3e', trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).half().cuda()
        return cls(
            model=model,
            tokenizer=tokenizer
        )

class Chatglm26b(LLM):
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

        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


    
    @classmethod
    def from_llm(
        cls,
    ):
        """Initialize the BabyAGI Controller."""
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
        model = model.eval()
                
        return cls(
            model=model,
            tokenizer=tokenizer
        )

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


# 使用chain
# prompt_template = '''
# {content}
# 帮助我分析上述公众舆论的原始内容。对于以下每个问题，请先提供推理过程，然后给出结论。
# 1.
# 问题：确定公众舆论的情感是积极的、中立的还是消极的。请注意，答案应限定为积极、中立或消极。
# 推理过程：分析内容中使用的语调、语境和表达方式，评估整体情感。
# 2.
# 问题：确定公众舆论所属的省份。
# 推理过程：分析内容中与特定地点有关的参考、人口统计或任何相关信息，以确定相关的省份。请注意，选择应限定为以下省份：北京、天津、河北、山西、内蒙古、辽宁、吉林、黑龙江、上海、江苏、浙江、安徽、福建、江西、山东、河南、湖北、湖南、广东、广西、海南、重庆、四川、贵州、云南、西藏、陕西、甘肃、青海、宁夏、新疆、台湾、香港、澳门或不确定。
# 3.
# 问题: 从公众舆论中提取3个关键词。你需要输出这些关键词。
# 推理过程：确定概括公众舆论的主题、主旨或焦点的重要术语或短语。
# 4.
# 问题：确定公众舆论涉及的业务细节。请用一句话回答。
# 推理过程：分析内容，识别与公司、产品、服务或其他与业务活动相关的任何相关信息。
# 5.
# 问题：确定公众舆论对用户的影响。请用一句话回答。
# 推理过程：评估公众舆论可能对个人、群体或社区产生的潜在影响、影响或后果。
# 6.
# 问题：确定公众舆论是否构成重大公众舆论。请注意，答案必须限定为“是，它构成了重大公众舆论”或“否，它构成了重大公众舆论”。
# 推理过程：评估公众舆论的规模、影响范围、重要性和重要性，以确定其作为重大舆论还是次要舆论的分类。
# 7.
# 问题：确定公众舆论是否需要人工跟进和解决。请注意，答案必须限定为“是”或“否”。
# 推理过程：评估公众舆论的性质、严重程度和紧迫性，以确定是否需要人工干预或解决。
# '''



def get_model_output(llm,content):
    prompt_template = '''
    {content}
    帮助我分析上述公众舆论的原始内容，你需要从以下几个方面进行分析，不需要重新输出问题，分别给出每个问题的回答和推理过程即可。
    1. 根据公众舆论中使用的语境和表达方式判断公众舆论的情感是积极的、中立的还是消极的？你只能选择"积极"、"消极"、"中立"中的一项作为答案。
    2. 根据公众舆论中与特定地点有关的参考相关信息，以确定相关的省份公众舆论属于哪个省份？需要特别注意的是只能选择如下的省份：北京、天津、河北、山西、内蒙古、辽宁、吉林、黑龙江、上海、江苏、浙江、安徽、福建、江西、山东、河南、湖北、湖南、广东、广西、海南、重庆、四川、贵州、云南、西藏、陕西、甘肃、青海、宁夏、新疆、台湾、香港、澳门或不确定。
    3. 根据公众舆论的主题、主旨或焦点确定公众舆论中的关键词是什么？
    4. 公众舆论涉及哪些业务细节？如果没有的话，请说明没有。
    5. 公众舆论对用户的有哪些影响？
    6. 根据公众舆论的规模、影响范围和重要性确定公众舆论是否构成重大公众舆论？
    7. 根据公众舆论的性质、严重程度和紧迫性确定公众舆论是否需要人工跟进和解决？ 
    最后，请总结分析结果。
    '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain.predict(content = content)


# 解析模型输出结果text
def get_final_results(text):
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
        result[tasks[i]] = content
    return result 



    # 4. 将结果写到文件中
    with open("resultschain.txt", "w") as file:
        file.write(json.dumps(results, ensure_ascii=False, indent=4))
    

def start():
    start_time = time.time()

    # 初始化模型
    llm = Vicuna7b.from_llm()
    # llm = OpenAI(openai_api_key = "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",temperature=0)
    # 获取舆情内容
    contents = get_data()
    # n = len(contents)
    n = 3
    results = []
    for i in range(0, n):
        result = {}
        result['content'] = contents[i]
        print(f'----------------------process {i+1} / {n}------------------------------\n')
        isRelated = judge_relation(contents[i])
        if 'yes' in isRelated.lower():
            result['isRelated'] = '相关'
            llm_output = get_model_output(llm,contents[i])
            print(llm_output)
            result['analysis'] = get_final_results(llm_output)
            results.append(result)
        else: 
            result['isRelated'] = '不相关'
            result['analysis'] = ''
        print(result)
    print(f'-------------------------process end--------------------------------\n')    

    # 4. 将结果写到文件中
    with open("chineseresults.txt", "w") as file:
        file.write(json.dumps(results, ensure_ascii=False, indent=4))

    end_time = time.time()
    print(f'执行该程序耗费的时间是{(end_time-start_time)/60}分')
    
start()