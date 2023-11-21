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
from typing import Optional
import chainlit as cl
import torch
import sys

# load user defined utils
sys.path.append('src/utils/')
from conversation_utils import create_prompt, get_response_from_qa_chain, answering_bot
from load_data import pdf_to_text, doc_to_text, write_text
from doc_to_embeddings import text_to_embeddings
from model_utils import load_llama2_llm


# environment for the app
# conda activate llama2-chainlit
# command to run the app
# chainlit run src/apps/localLLM_withRAG.py --port 8001 -w

rawdata_path = 'data/rawdata/'
modelpath = "models/llama-2-7b-chat.Q2_K.gguf"


credentials = {"admin": "admin",
                  "amit.jha": "amit.jha"}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database    
  
  if username == "admin":
    if (username, password) == ("admin", "admin"):
        return cl.AppUser(username="admin", role="ADMIN", provider="credentials")
    else:
        return None
    
  if username == "amit.jha":
      if (username, password) == ("amit.jha", "amit.jha"):
        return cl.AppUser(username="amit.jha", role="user", provider="credentials")
      else:
        return("Please provide the correct username and password")


# Loading the local model into LLM
llm = load_llama2_llm(modelpath)

    
@cl.on_chat_start
async def factory():
    
    # loads the data by the user
    files = None

    # wait for the user to upload a data file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text or pdf file to begin with!", 
                     accept={"text/csv": [".txt"], "application/pdf": [".pdf"], 
                                "text/doc": [".docx", ".doc"]},
                     max_size_mb=2
        ).send()
        
    file = files[0]
    
    filetype = file.name.split('.')[1]
    print(file.name.split('.'), filetype)
    
    if filetype == 'pdf':
        # reads and convert pdf data to text 
        texts = pdf_to_text(file)
#     elif filetype == 'docx' or filetype == 'doc':
#         # reads and convert pdf data to text 
#         texts = doc_to_text(file)
    else:
        print("Uploaded file type: {0} not supported currently. Please upload supported(text/pdf/doc) filetype", filetype)
    # write the raw texts to file
    write_text(rawdata_path, texts)
    
    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{files[0].name}` is uploaded, we are processing the document!"
    ).send()
    
    # create embeddings for the uploaded documents
    db_faiss_path = text_to_embeddings(texts)
        
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
