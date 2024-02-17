from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

pdfs = "./pdfs/"
output = "vectorstore/db_faiss"

def create_vector_db():
    loader = DirectoryLoader(pdfs,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(output)

create_vector_db()