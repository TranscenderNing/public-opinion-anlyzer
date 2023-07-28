# Import Chat completion template and set-up variables
import openai
import urllib.parse

openai.api_key = "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" # Key is ignored and does not matter
openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"
# Alternate mirrors
# openai.api_base = "http://34.132.127.197:8000/v1"

# Report issues
def raise_issue(e, model, prompt):
    issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
    issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
    issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
    print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")

# Query Gorilla server
def get_gorilla_response(prompt="I would like to translate from English to French.", model="gorilla-7b-hf-v1"):
  try:
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}]
    )
    # print(completion)
    return completion.choices[0].message.content
  except Exception as e:
    raise_issue(e, model, prompt)
        
        
# prompt = "Evaluating the sentiment of public opinion. (positive or negative or neutral)"
prompt = "You are a specialized robot in extracting keywords from Chinese sentences. Given a Chinese sentence, you provide the keywords. The sentence is: '6月5日，在第二届移动网络高质量发展论坛上，工信部发布了2022年全国重点场所移动网络质量评测排名，中国移动浙江公司荣获“重点区域移动网络质量卓越运营商”称号 '"
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))