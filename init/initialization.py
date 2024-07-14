import os
from dotenv import load_dotenv

from huggingface_hub import login
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from constants import constants
from init import document_preparation


def initial_setup(dataset):
    vector_store = _setup_vector_store(dataset)
    retriever = _setup_retriever(vector_store)
    llm = _setup_llm()

    return [llm, retriever]


def _setup_vector_store(dataset):
    modified_chunks = document_preparation.setup_documents(dataset)

    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    # print_to_console.print_vector_store_content(vector_store)

    return vector_store


def _setup_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type=constants.SEARCH_TYPE_VALUE,
                                          search_kwargs={constants.K_KEY: 4, constants.SCORE_THRESHOLD_KEY: 0.5})

    return retriever


def _setup_llm():
    load_dotenv()
    hugging_face_api_key = os.getenv(constants.HUGGING_FACE_API_KEY)
    login(hugging_face_api_key)

    llm = HuggingFaceEndpoint(
        repo_id=constants.LLM_MODEL_NAME,
        huggingfacehub_api_token=hugging_face_api_key,
        task=constants.LLM_TASK,
        max_new_tokens=512,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.1
    )

    return ChatHuggingFace(llm=llm)