from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import LlamaCPP
from datetime import datetime
import sys

# conda environment for this app - 
# conda activate llm
# command to run this app
# python3 src/apps/localLLM_cmtPrompt.py -w

n_batch = 256
n_gpu_layers = 40
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelpath = "models/llama-2-7b-chat.Q2_K.gguf"


llm = LlamaCPP(
        model_path = modelpath,
        temperature=0.1,
        max_new_tokens=3000,
        
        context_window=3900,
        generate_kwargs = {},
        model_kwargs = {"n_gpu_layers": n_gpu_layers},
        messages_to_prompt = messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )


# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    start_time = datetime.now()
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting")
        sys.exit()

    response = llm.complete(query)
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
    print('Answer: ' + response.text + '\n')
    chat_history.append((query, response.text))

