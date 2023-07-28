from langchain import HuggingFaceHub,HuggingFacePipeline
import os
# os.environ['CURL_CA_BUNDLE'] = ''

llm = HuggingFaceHub(
    repo_id="lmsys/vicuna-7b-v1.3",
    model_kwargs={"temperature": 0, "max_length": 64},
    huggingfacehub_api_token = "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"
)



# llm = HuggingFacePipeline.from_model_id(
#     model_id="bigscience/bloom-1b7",
#     task="text-generation",
#     model_kwargs={"temperature": 0, "max_length": 64},
# )


from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is electroencephalography?"

print(llm_chain.run(question))


