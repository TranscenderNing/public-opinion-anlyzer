
# gorilla生成的代码  情感分类
# from transformers import pipeline

# def load_model():
#     classifier = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
#     return classifier

# def process_data(text, classifier):
#     response = classifier(text)
#     return response[0]['label']

# text = 'Evaluating the sentiment of public opinion.'
# # Load the model
# classifier = load_model()

# # Process the data
# response = process_data(text, classifier)

# print(response)



## gorilla生成的代码 提取关键词

# from keybert import KeyBERT

# doc = """
# On June 5th, at the Second Mobile Network High-Quality Development Forum, the Ministry of Industry and Information Technology (MIIT) released the ranking of national key venue mobile network quality assessment for the year 2022. China Mobile Zhejiang Company was awarded the title of "Excellent Operator in Key Area Mobile Network Quality
#       """
# kw_model = KeyBERT()
# keywords = kw_model.extract_keywords(doc)
# print(keywords)


from transformers import AutoTokenizer, AutoModel
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    model = AutoModel.from_pretrained('GanymedeNil/text2vec-large-chinese')
    return tokenizer, model

def process_data(sentence, tokenizer, model):
    tokens = tokenizer.encode(sentence, return_tensors='pt')
    embeddings = model(tokens)[0].mean(axis=1)
    keywords = torch.topk(embeddings, k=5, dim=1).indices.squeeze(0).tolist()
    response = [tokenizer.decode([keyword]) for keyword in keywords]
    return response

sentence = '6月5日,在第二届移动网络高质量发展论坛上,工信部发布了2022年全国重点场所移动网络质量评测排名,中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号 '

# Load the model and tokenizer
tokenizer, model = load_model()

# Process the data
response = process_data(sentence,tokenizer,model)
print(response)
