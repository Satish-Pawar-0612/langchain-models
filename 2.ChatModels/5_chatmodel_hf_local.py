from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# These next two lines are optional, but they can be used to set the cache directory for Hugging Face models. If you want to use a custom cache directory, uncomment the following lines and set the path accordingly.
# import os
# os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)