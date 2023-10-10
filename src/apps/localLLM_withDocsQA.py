from llama_index.callbacks.base import CallbackManager
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.llms import CTransformers
from datetime import datetime
import chainlit as cl
import torch
import sys

sys.path.append('src/utils/')

from conversation_utils import create_prompt, get_response_from_qa_chain, answering_bot
from doc_to_embeddings import pdf_to_embeddings

# environment for the app
# conda activate llama2-chainlit
# command to run the app
# chainlit run src/apps/localLLM_withDocsQA.py --port 8001 -w


n_batch = 256
n_gpu_layers = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
docpath = "data/rawdata/report.pdf"
modelpath = "models/llama-2-7b-chat.Q2_K.gguf"


@cl.cache
def load_llama2_llm(modelpath):
    # load the model that was downloaded locally
    llm = CTransformers(
              model = modelpath,
              
              model_type="llama",
              max_new_tokens=512,
              n_batch = n_batch,
              n_gpu_layers=n_gpu_layers,
              temperature=0.4,
              max_length=3000,
              device_map="cpu"
    )
    return(llm)

# Loading the local model into LLM
llm = load_llama2_llm(modelpath)

    
@cl.on_chat_start
async def factory():
    # create embeddings for the uploaded documents
    db_faiss_path = pdf_to_embeddings(docpath)
    
    print(db_faiss_path)
    
    # create llm chain for RAG usecase
    chain = answering_bot(llm, db_faiss_path)
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "NBI Assistant is ready. Ask questions on the documents indexed?"
    await msg.update()
    
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    start_time = datetime.now()
    chain = cl.user_session.get("chain")
    response = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
    
    await cl.Message(content=response["result"]).send()
