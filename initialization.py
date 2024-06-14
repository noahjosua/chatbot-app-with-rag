import os
from dotenv import load_dotenv

from huggingface_hub import login
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

import constants
from helper import print_to_console
import document_preparation


def initial_setup():
    vector_store = _setup_vector_store()
    retriever = _setup_retriever(vector_store)
    llm = _setup_llm()

    return [llm, retriever]


def _setup_vector_store():
    modified_chunks = document_preparation.setup_documents()

    embeddings = SentenceTransformerEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    print_to_console.print_vector_store_content(vector_store)

    return vector_store


def _setup_retriever(vector_store):
    # TODO NOAH constants
    retriever = vector_store.as_retriever(search_type='similarity_score_threshold',
                                          search_kwargs={constants.K_KEY: 4, 'score_threshold': 0.5})

    return retriever


def _setup_llm():
    load_dotenv()
    hugging_face_api_key = os.getenv(constants.HUGGING_FACE_API_KEY)
    login(hugging_face_api_key)

    llm = HuggingFaceEndpoint(
        repo_id=constants.LLM_MODEL_NAME,
        huggingfacehub_api_token=hugging_face_api_key,
        task=constants.LLM_TASK,  # question-answering
        max_new_tokens=512,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.1
    )

    return ChatHuggingFace(llm=llm)
