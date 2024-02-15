from langchain_community.llms import CTransformers

llm=CTransformers(model="./llama-2-7b-chat.ggmlv3.q8_0.bin",
                   model_type="llama",config={
                       "max_new_tokens":256,
                       "temperature":"0.01"
                   })
response=llm("Hello")
print(response)