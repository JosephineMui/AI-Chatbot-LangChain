import os

from prompt_toolkit import prompt import torch

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_huggingface import ( HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace )

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

def init_llm():
    global llm_hub, embeddings

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR API KEY"

    base_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        temperature=0.1,
        max_new_tokens=600,
    )

    # wrap with ChatHuggingFace
    llm_hub = ChatHuggingFace(llm=base_llm)

    embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}, )

def process_document(document_path):
    global conversation_retrieval_chain

    loader = PyPDFLoader(document_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1024, chunk_overlap=64 )
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embedding=embeddings)

    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub, chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False, input_key="question")

def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history

    output = conversation_retrieval_chain.invoke( {"question": prompt, "chat_history": chat_history} )
    answer = output["result"]

    chat_history.append((prompt, answer))
    return answer.strip()

init_llm()