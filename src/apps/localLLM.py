from langchain import PromptTemplate,  LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers
import chainlit as cl
from datetime import datetime
import torch

import os

# command to run
# chainlit run src/apps/localLLM.py -w

template = """
           Question: {question}
           
           Answer: Let's think steps by step.           
           """

template = """You are an helpful computer AI agent and your name is NBI Assitant. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in a step by step manner, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 

Question: {question}

Response for Questions asked.
answer:
"""

n_batch = 256
n_gpu_layers = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelpath = "models/llama-2-7b-chat.Q2_K.gguf"


@cl.cache
def load_llama2_llm():
    # load the model that was downloaded locally
    
    llm = CTransformers(
              model = modelpath,
              model_type="llama",
              max_new_tokens=512,
              context_length=3000,
#               n_batch = n_batch,
#               n_gpu_layers=n_gpu_layers,
              temperature=0.4,
              max_length=3000,
              device_map="cpu"
    )
    return(llm)

llm = load_llama2_llm()

@cl.on_chat_start
def main():
    # Instantiate the chain for the user session
    prompt = PromptTemplate(template = template, input_variables=["question"])
    llm_chain = LLMChain(prompt = prompt, llm = llm, verbose=True)
    
    
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
