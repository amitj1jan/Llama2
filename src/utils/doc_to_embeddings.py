from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
indexpath = "data/vectorstore/"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

def text_to_embeddings(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
#     text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size = 1000,
#                         chunk_overlap  = 20,
#                         length_function = len,
#                         is_separator_regex = False,
#                         )

    whole_text = text_splitter.create_documents([texts])

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                      model_kwargs={'device': device})
    faiss_index = FAISS.from_documents(whole_text, embeddings)
    faiss_index_path = indexpath + 'temp-index'
    faiss_index.save_local(faiss_index_path)
    return(faiss_index_path)
    






