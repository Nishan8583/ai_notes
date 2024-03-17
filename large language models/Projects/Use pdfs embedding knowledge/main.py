from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.globals import set_debug


pdfs = "./pdfs/"
output = "vectorstore/db_faiss"

def create_vector_db():
    loader = PyPDFLoader("Claire Agutter - ITIL 4 essentials _ your essential guide for the ITIL 4 Foundation exam and beyond (2020) - libgen.li.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("starting cuda")
    # just a random embedding i chose for the project
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
    
    # tokenize the texts, and storing in FIASS db locally
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(output)


def create_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question about ITIL 4.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

context: {context}
Question: {question}
Only return the helpful answer and nothing else.
Helpful answer:
"""
    prompt=PromptTemplate(template=custom_prompt_template,
                          input_variables=['context','question'])
    return prompt

def create_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_module():

    # loading the metas llama LLM
    print("Loading LLM")
    llm = CTransformers(
        model="llama-2-7b-chat.Q8_0.gguf",
        model_type="llama",
        max_new_token=2048,
        temperature=0.0,
        gpu_layers=50,
    )
    print("LLM Loaded")
    print("Loading embeddings")
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cuda'}
    )
    db = FAISS.load_local(output,embeddings,allow_dangerous_deserialization=True)
    print("embeddings DB loaded")

    qa=create_qa_chain(llm,create_prompt(),db)
    print("custom chain created")
    return qa

def main():
    set_debug(True)
    llm = load_module()
    user_input=input("you >>> ")
    user_input = user_input.lower()
    while user_input != "quit" or user_input != 'q':
        print("Ur query", user_input)
        response = llm({"query":user_input})
        print("LLM Response >>> {}".format(response["result"]))
        user_input=input("you >>> ")
        user_input = user_input.lower()
   #create_vector_db()

main()
#create_vector_db()