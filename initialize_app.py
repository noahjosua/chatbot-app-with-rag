import os  # Operating system functionalities
from dotenv import load_dotenv  # Environment variable handling

# Langchain #
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

from huggingface_hub import login

import helper
from helper import *


def initialize_app():
    load_dotenv()
    hugging_face_api_key = os.getenv('HUGGING_FACE_API_KEY')

    # load documents
    loader = CSVLoader(file_path='netflix_titles.csv', encoding='utf-8')
    documents = loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    modified_chunks = helper.modify_metadata(chunks)

    # set up embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2') # api_key=hugging_face_api_key

    # set up vector db
    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    # print(vector_store.index.ntotal)

    # set up retriever
    retriever = vector_store.as_retriever() # By default, the vector store retriever uses similarity search

    # set up LLM
    login(hugging_face_api_key) # TODO verstehen, wozu das gebraucht wird
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        huggingfacehub_api_token=hugging_face_api_key,
        task='text-generation',
        max_new_tokens=512,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.1
    )

    return [ChatHuggingFace(llm=llm), retriever]
