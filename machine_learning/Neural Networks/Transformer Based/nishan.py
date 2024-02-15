
from langchain.prompts import Prompt
from langchain.llms import CTransformers

## Function To get response from LLAma 2 model
def getLLamaresponse():

    ### LLama2 model
    llm=CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':800,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""Tell me what SQL vulnerability with a common example"""
    
    #prompt= Prompt(template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(template)
    print(response)
    print(len(response))
    return response

getLLamaresponse()