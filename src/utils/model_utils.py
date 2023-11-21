from langchain.llms import CTransformers
import chainlit as cl
import torch

n_batch = 256
n_gpu_layers = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {'max_new_tokens': 512, 'context_length': 4096,         
            'gpu_layers': n_gpu_layers,'batch_size': n_batch,   
            'temperature': 0.4
         }

@cl.cache
def load_llama2_llm(modelpath):
    # load the model that was downloaded locally    
    llm = CTransformers(
              model = modelpath,
              model_type="llama",
              config = config,   
              device_map= device
    )
    return(llm)
