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
import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import gradio as gr
import argparse
import warnings
import os
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import time


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/model/13B_hf")
parser.add_argument("--lora_path", type=str, default="checkpoint-3000")
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()
print(args)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# 自定义模型 vicuna7b
class Vicuna7b(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    
    def generate_prompt(self,instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""


    def evaluate(
        self,
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        min_new_tokens=1,
        repetition_penalty=2.0,
        **kwargs,
    ):
        prompt = self.generate_prompt(input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = "cuda"
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
            min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
            **kwargs,
        )
        with torch.no_grad():
            if args.use_typewriter:
                for generation_output in self.model.stream_generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    repetition_penalty=float(repetition_penalty),
                ):
                    outputs = self.tokenizer.batch_decode(generation_output)
                    show_text = "\n--------------------------------------------\n".join(
                        [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
                    )
                    # if show_text== '':
                    #     yield last_show_text
                    # else:
                    yield show_text
                yield outputs[0].split("### Response:")[1].strip().replace('�','')
            else:
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    repetition_penalty=1.3,
                )
                output = generation_output.sequences[0]
                output = self.tokenizer.decode(output).split("### Response:")[1].strip()
                print(output)
                yield output



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
        input = "Answer the user query.\n    The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{\"properties\": {\"sentiment\": {\"title\": \"Sentiment\", \"description\": \"answer whether the sentiment positive or negative. \", \"type\": \"string\"}, \"province\": {\"title\": \"Province\", \"description\": \"answer to which province is the sentiment associated with.\", \"type\": \"string\"}, \"keyword\": {\"title\": \"Keyword\", \"description\": \"The keywords related to the sentiment. Maybe more than one.\", \"type\": \"string\"}, \"impactOnUsers\": {\"title\": \"Impactonusers\", \"description\": \"how does the sentiment impact the user? As detailed as possible.\", \"type\": \"string\"}, \"isMajor\": {\"title\": \"Ismajor\", \"description\": \" Whether it is a major sentiment and explain why. As detailed as possible.\", \"type\": \"string\"}, \"manualFollow\": {\"title\": \"Manualfollow\", \"description\": \"Whether it require manual follow-up and explain why. As detailed as possible.\", \"type\": \"string\"}}, \"required\": [\"sentiment\", \"province\", \"keyword\", \"impactOnUsers\", \"isMajor\", \"manualFollow\"]}\n```\n    user query: Analyze the given sentiment content. provide a paragraph output of the sentiment analysis results, including the following aspects: whether the sentiment is positive or negative, the province associated with the sentiment, the keywords related to the sentiment, the impact of the sentiment on users, whether it is a major sentiment, and whether manual follow-up is required. Every aspect needs to be as detailed as possible. Sentiment content:6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号"
        prompt = self.generate_prompt(input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
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
        output = self.tokenizer.decode(output).split("### Response:")[1].strip()
        # print(output)

        return output

    @classmethod
    def from_llm(
        cls
    ):     
        # tokenizer 
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

        # model
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                BASE_MODEL,
                load_in_8bit=LOAD_8BIT,
                torch_dtype=torch.float16,
                device_map="auto", #device_map={"": 0},
            )
            model = StreamPeftGenerationMixin.from_pretrained(
                model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map="auto", #device_map={"": 0}
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = StreamPeftGenerationMixin.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = StreamPeftGenerationMixin.from_pretrained(
                model,
                LORA_WEIGHTS,
                device_map={"": device},
            )


        if not LOAD_8BIT:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)


        return cls(
            model=model,
            tokenizer=tokenizer
        )


# 从json文件中获取内容
def get_data():
    # 打开并读取 JSON 文件
    with open('/home/ldn/langchain/vicunaModel/datasets', 'r') as file:
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


# 4. 将结果写到文件中
with open("results.txt", "w") as file:
    # 遍历字典的键值对，将其写入文件
    idx = 1
    for key, value in results.items():
        file.write(f"热点舆情内容{idx}:{key}\n分析结果:\n{value}\n\n")
        idx += 1
    
end_time = time.time()
print(f'执行该程序耗费的时间是{end_time-start_time}秒')






