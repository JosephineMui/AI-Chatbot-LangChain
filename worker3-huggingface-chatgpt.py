from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA

import os


# -----------------------------------------
# 1 Load PDF
# -----------------------------------------
pdf_path = "data/document.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()


# -----------------------------------------
# 2 Split into chunks
# -----------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)


# -----------------------------------------
# 3 Create HuggingFace Embeddings
# -----------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------------
# 4 Create / Load Chroma Vector Database
# -----------------------------------------
persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_directory
)

vectordb.persist()


# -----------------------------------------
# 5 Setup HuggingFace LLM
# -----------------------------------------
# Requires HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_TOKEN"

llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm_endpoint)


# -----------------------------------------
# 6 Create Retriever
# -----------------------------------------
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# -----------------------------------------
# 7 Create RAG QA Chain
# -----------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=True
)


# -----------------------------------------
# 8 Chatbot Loop
# -----------------------------------------
print("\nPDF Chatbot Ready. Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    result = qa_chain.invoke({"query": question})

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print(doc.metadata)

    print("\n-----------------------------------\n")