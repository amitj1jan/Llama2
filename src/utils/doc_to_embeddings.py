from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

indexpath = "data/vectorstore/"

def pdf_to_embeddings(docpath):
    loader = PyPDFLoader(docpath)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    faiss_index = FAISS.from_documents(texts, embeddings)
    faiss_index_path = indexpath + 'temp-index'
    faiss_index.save_local(faiss_index_path)
    return(faiss_index_path)
    






