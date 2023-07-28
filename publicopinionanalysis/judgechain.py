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
llm = Vicuna7b.from_llm()

# judge_template = "Please help me determine whether the following public opinion is related to China Mobile company. If it is related, you need to output true; otherwise, you need to output false. {content}"
judge_template = '''Answer the following questions as best you can. 
Answer should be one of [yes, no].
Question : China Mobile Communications Group Co., Ltd., also known as China Mobile Group, is a large-scale state-owned enterprise engaged in the telecommunications industry in the People's Republic of China. China Mobile provides comprehensive communication services across all 31 provinces, autonomous regions, municipalities, and the Hong Kong Special Administrative Region in mainland China. Its primary business includes mobile voice and data services, wired broadband, as well as other communication and information services.Does the text involve China Mobile Communications Group Co., Ltd.? 
Text : {content}'''
judge_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(judge_template)
)

# content = "7月18日，周杰伦演唱会天津站开票。据大麦平台显示，本场天津演唱会大麦单平台放票近13万张，共分四轮开票。超520万人标记“想看”。经实际抢票发现，四轮开票都在30秒内售罄。网友吐槽票不好抢的同时，黄牛已将票炒至天价。据证券时报，有黄牛称内场前三排售价高至19800元。"
# content = "7月19日，安徽利辛通报 “执法船只载多名女性游玩”一事。经查，视频船只系利辛县水政执法大队所属水政船。15日，县水政执法大队张某某驾船巡河，私自搭载朱某某及4名水草清理女子，几人沿河道查看水面杂草清理情况，违反县水政执法大队船只管理规定。经研究决定给予张某某通报批评，扣发3个月绩效工资，调离原工作岗位。给予县水政执法大队主要负责人李某通报批评。"
content = '''最近，不少市民收到移动公司10086的短信或电话，要求进行手机“实名认证”，这是怎么回事？对此，中国移动通信集团湖南有限公司发布公告称，为坚决遏制电信网络诈骗案件高发势头，维护人民群众切身利益，根据湖南省公安厅断卡行动最新要求，该公司将于近期分批次提供二次实名认证服务。“如您或亲友接收到10086、10085发送的实名认证提醒短信或来电，请您及时按照提醒步骤完成认证。”收到短信或电话，请市民在“湖南移动微厅”公众号对话框输入“11”进行线上认证，或携带本人手机卡及身份证件原件到入网地移动网点实名认证'''
print(judge_chain.predict(content = content))

# def judge_relation(content):
#     key_list = ["中国移动","移动公司","移动网络","10086"]
#     for key in key_list:
#         if key in content:
#             return "true"
#     return "false"

# print(judge_relation(content))
    
    
