from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

custom_prompt_template = """You are an helpful computer AI agent and your name is NBI Assitant. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in a step by step manner, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""


def create_prompt(prompt_template):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    
    prompt = PromptTemplate(template=prompt_template,
                           input_variables=['context', 'question'])
    return(prompt)


# retrieval Chain
def get_response_from_qa_chain(llm, prompt, db):
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    
    return(retrieval_chain)


# asnwering bot creation
def answering_bot(llm, db_faiss_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cuda'})
    
    vectorstore = FAISS.load_local(db_faiss_path, embeddings)
    message_prompt = create_prompt(custom_prompt_template)
    response = get_response_from_qa_chain(llm, message_prompt, vectorstore)
    return(response)
