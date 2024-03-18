
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import ConversationChain
import argparse
from langchain.globals import set_debug
set_debug(True)

def load_module_gpu(model_path):

    # loading the metas llama LLM
    print("Loading LLM")
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_token=6000,
        temperature=0.0,
        config={"gpu_layers":50}, # Now GPU works, yipeeee
    )
    print("LLM Loaded")

    prompt_template = """<s>[INST] <<SYS>>
{{ You are a helpful AI Assistant}}<<SYS>>
###


{{{input}}}[/INST]

"""
    #prompt = PromptTemplate(template=prompt_template,input_variables=["input"])

    #qa_chain = ConversationChain(llm,prompt)
    #print("custom chain created")
    #return qa_chain
    return llm
#
def main():
    parser =argparse.ArgumentParser("A simple code scanner that uses llama to scan for code")
    parser.add_argument("--file","-f",default=None,help="path to code to scan")
    arguements = parser.parse_args()
    if arguements.file == None:
        print(parser.print_help())
        return
    code = ""
    with open(arguements.file,"r") as f:
        code = f.read()
    code = "Please find vulnerability in the following code: ```{}```".format(code)
    llm = load_module_gpu("llama-2-7b-chat.Q8_0.gguf")
    print(llm(code))
#    print(llm({'query':"what is malware analysis?"}))
#    print(llm({'query':"How to treat tounge infection disease?"}))
    #print(llm({'query':"Why are stars so bright?"}))
   #create_vector_db()

main()