from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from datetime import datetime
from typing import Optional
import sys
import os

# load user defined utils
sys.path.append('src/utils/')
from conversation_utils import create_prompt, get_response_from_qa_chain, answering_bot
from model_utils import load_llama2_llm

# command to run
# chainlit run src/apps/contextAware_localLLM.py -w

prompt_template = """You are an helpful computer AI agent and your name is NBI Assitant. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in a step by step manner, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{chat_history}
Question: {question}

Response for Questions asked.
answer:
"""
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

path = "models/"
model = "llama-2-7b-chat.Q2_K.gguf"
modelpath = "models/" + model
chat_history = ""


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
def main():
    # Instantiate the chain for the user session
    prompt = PromptTemplate(template = prompt_template, 
                            input_variables=["chat_history", "question"])
    chat_history=""
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")
    llm_chain = LLMChain(llm = llm, 
                         prompt = prompt,  
                         verbose=True,
                         memory = memory)
    
    
#     response = llm_chain.run(["Who is the Pope ?"])
#     print(response)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)
                            

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    start_time = datetime.now()
    llm_chain = cl.user_session.get("llm_chain")
                            
    # Call the chain asynchronously
    response = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
    
    await cl.Message(content = response["text"]).send() 
