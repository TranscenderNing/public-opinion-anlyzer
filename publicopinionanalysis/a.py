
# # # from googletrans import Translator

# # # text = '''The public opinion does not necessarily require manual follow-up and resolution, but it may require further investigation and analysis to understand the implications of the recognition for China Mobile and its customers.
# # # '''
# # # translator = Translator(service_urls=['translate.google.cn'])
# # # result = translator.translate(text, dest='zh-cn').text
# # # print(result)


# # # from translate import Translator

# # # # 英语翻译中文
# # # translator = Translator(to_lang="chinese")
# # # # translation = translator.translate("The public opinion does not necessarily constitute a major public opinion, but it may be significant for the local market and mobile network service users in Zhejiang Province.")
# # # translation = translator.translate('hello world')
# # # print(translation)


# # # from googletrans import Translator
# # # # 设置Google翻译服务地址
# # # translator = Translator(service_urls=[
# # #       'translate.google.com'
# # # ])

# # # translator.raise_Exception = True

# # # translation = translator.translate('The public opinion does require manual follow-up and resolution, as it involves legal issues and potential criminal behavior.', dest='zh-CN')
# # # print(translation.text)


# #     # def load_model(self):
# #     #     translation_pipeline = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
# #     #     return translation_pipeline

# #     # def process_data(self,input_text, translation_pipeline):
# #     #     response = translation_pipeline(input_text)
# #     #     translated_text = response[0]['translation_text']
# #     #     return translated_text
# # from transformers import pipeline
# # from typing import Any

# # # class TranslateEn2Zh():
# # #     def __init__(self):
# # #         self.translation_pipeline = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
        
# # #     def process_data(self,input_text):
# # #         response = self.translation_pipeline(input_text)
# # #         translated_text = response[0]['translation_text']
# # #         return translated_text
   
 
# # # input_text = "我是一名歌手"
# # # translator = TranslateEn2Zh()

# # # response = translator.process_data(input_text)
# # # print(response)



# # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # # def load_model():
# # #     tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
# # #     model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
# # #     return tokenizer, model

# # # def process_data(input_text, tokenizer, model):
# # #     input_tokens = tokenizer.encode(input_text, return_tensors='pt')
# # #     translated_tokens = model.generate(input_tokens)
# # #     response = tokenizer.decode(translated_tokens[:, input_tokens.shape[-1]:][0], skip_special_tokens=True)
# # #     return response

# # # input_text = "7月15日，根据《中央企业负责人经营业绩考核办法》，国务院国资委公布了2022年度中央企业负责人经营业绩考核结果，中国移动再次获评A级企业，连续19个年度获得中央企业负责人经营业绩考核A级企业"

# # # tokenizer, model = load_model()
# # # print(process_data(input_text,tokenizer, model))

# # # import re

# # # string = '这是一个包含"双引号内容","dfad",的字符"fdafsda"串'

# # # # 使用正则表达式匹配双引号中的内容
# # # matches = re.findall(r'"([^"]*)"', string)

# # # # 输出匹配到的内容
# # # for match in matches:
# # #     print(match)



# # provinces = [
# #     "北京市",
# #     "天津市",
# #     "河北省",
# #     "山西省",
# #     "内蒙古自治区",
# #     "辽宁省",
# #     "吉林省",
# #     "黑龙江省",
# #     "上海市",
# #     "江苏省",
# #     "浙江省",
# #     "安徽省",
# #     "福建省",
# #     "江西省",
# #     "山东省",
# #     "河南省",
# #     "湖北省",
# #     "湖南省",
# #     "广东省",
# #     "广西壮族自治区",
# #     "海南省",
# #     "重庆市",
# #     "四川省",
# #     "贵州省",
# #     "云南省",
# #     "西藏自治区",
# #     "陕西省",
# #     "甘肃省",
# #     "青海省",
# #     "宁夏回族自治区",
# #     "新疆维吾尔自治区",
# #     "台湾省",
# #     "香港特别行政区",
# #     "澳门特别行政区"
# # ]

# # provinces_str = ",".join(provinces)

# # print(provinces_str)






# # from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
# # # tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-x-en")

# # # # translate German to English
# # # tokenizer.src_lang = "zh"
# # # inputs = tokenizer("【连续19年！A级】7月15日，根据《中央企业负责人经营业绩考核办法》，国务院国资委公布了2022年度中央企业负责人经营业绩考核结果，中国移动再次获评A级企业，连续19个年度获得中央企业负责人经营业绩考核A级企业！", return_tensors="pt")
# # # generated_tokens = model.generate(**inputs)
# # # res =tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # # print(res)
# # # => "A model for many languages"


# # class TranslateZh2En():
# #     def __init__(self):
# #         self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
# #         self.tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
# #         self.tokenizer.src_lang = "zh"
        
# #     def process_data(self,input_text):
# #         inputs = self.tokenizer(input_text, return_tensors="pt")
# #         generated_tokens = self.model.generate(**inputs)
# #         res = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# #         return res
    

# # translator = TranslateZh2En()
# # res = translator.process_data("【连续19年！A级】7月15日，根据《中央企业负责人经营业绩考核办法》，国务院国资委公布了2022年度中央企业负责人经营业绩考核结果，中国移动再次获评A级企业，连续19个年度获得中央企业负责人经营业绩考核A级企业！")
# # print(res)


# import re
# # str = 'Manual follow-up and resolution required: Not specified\nReasoning process: The content does not discuss the need for manual intervention or resolution. However, it is important to monitor the potential effects and impacts of the news on users and the industry as a whole.\n'

# # pattern1 = r':\s(.+?)\nReasoning process:\s(.+?)\n'
# # matches1 = re.findall(pattern1, str, re.DOTALL)

# # print(len(matches1))


# # for match1 in matches1:
# #     conclusion = match1[0]
# #     reasoning_process = match1[1]
# #     print(conclusion)
# #     print(reasoning_process)
    
    
    
    
# text = '1. Sentiment: Positive\nReasoning process: The content describes the achievement of the China Mobile Zhejiang Company as the "Excellent Operator of Mobile Network Quality in Key Regions," which indicates a positive sentiment towards the company.\n2. Province: Zhejiang\nReasoning process: The content specifically mentions the China Mobile Zhejiang Company as the winner of the mobile network quality evaluation rankings in key regions.\n3. Keywords: mobile network, quality, evaluation, operator, Zhejiang\nReasoning process: The main subject of the public opinion is the quality evaluation of mobile networks, and the key operators involved are China Mobile and Zhejiang Company.\n4. Business details: N/A\nReasoning process: There is no relevant information about business details in the content.\n5. Impact on users: N/A\nReasoning process: The content does not provide any information about the potential effects or consequences on users.\n6. Classification: Major public opinion\nReasoning process: The content refers to the mobile network quality evaluation rankings of key regions in China, which is a significant and important issue for the public.\n7. Manual follow-up and resolution: N/A\nReasoning process: There is no indication in the content that manual follow-up or resolution is necessary.\n'
# # pattern = r"\d+\.\s+(.*)"
# # pattern = r'\d+\.\s+\d+\.'
# pattern = r'\d+\.\s(.*?)(?=\n\d+\.\s|$)'
# matches = re.findall(pattern, text)
# print(len(matches))
# # 判断分析是否完整
# if len(matches) > 0:
#     result = {}
#     tasks = ['sentiment', 'province', 'keywords', 'businessDetails', 'impactOnUsers', 'isMajor', 'manualFollowUp']
#     for i, value in enumerate(matches):
#         print(value)
#         # pattern1 = r'Conclusion:\s(.+?)\nReasoning process:\s(.+?)\n'
#         pattern1 = r':\s(.+?)\nReasoning process:\s(.+?)\n'
#         matches1 = re.findall(pattern1, value, re.DOTALL)
#         print(value)
#         print(len(matches1))
#         for match1 in matches1:
#             conclusion = match1[0]
#             print(conclusion)
#             reasoning_process = match1[1]
#             print(reasoning_process)


# # import re


# # text = '1. First sentence.\n2. Second sentence.\n3. Third sentence.\n4. Fourth sentence.\n5. Fifth sentence.\n'

# # pattern = r'\b\d+\.\s(.*?)(?=\n\d+\.\s|$)'

# # matches = re.findall(pattern, text)

# # for match in matches:
# #     print(match)



from transformers.pipelines import pipeline
class SentimentJudge():
    def __init__(self):
        self.classifier = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
        
    def process_data(self,text):
        response = self.classifier(text)
        return response[0]['label']
    
sentiment_judge = SentimentJudge()
res = sentiment_judge.process_data("5月24日报道，江苏连云港。举报人小佳称自己是移动公司的渠道老板（即合作营业厅负责人），刘某经常约她吃饭，她都没去，“2020年8月被他关了工号，不得不去赴约，随后发生了迷奸。")
print(res)