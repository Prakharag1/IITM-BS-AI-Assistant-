#lood chunking embedding store

import os 
from dotenv  import load_dotenv
from langchain_core.documents import Document 
from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader

print("1")

load_dotenv()

# os.environ["open"]
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("2")
DATA_PATH="../knowledge_base"
FAISS_PATH = "../faiss_index"
#loading data 
print("3")

url = "https://docs.google.com/document/u/2/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub#h.1fob9te"
loader = WebBaseLoader(url)
print("4")
documents = loader.load()

# pdf_loader=DirectoryLoader(DATA_PATH,glob="**/*.pdf",loader_cls=PyPDFLoader)
# print("4")
# documents=pdf_loader.load()

print("doc splitting....")
#splitting 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 150)
docs = text_splitter.split_documents(documents)

print("creating Embedddings....")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs,embeddings)
db.save_local(FAISS_PATH)
print("FAISS index created successfully..")
