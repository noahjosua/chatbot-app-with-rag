import os  # Operating system functionalities
from dotenv import load_dotenv  # Environment variable handling

# Langchain #
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

from huggingface_hub import login

import constants
import helper
import preprocess_dataset


def initial_setup():
    load_dotenv()
    hugging_face_api_key = os.getenv(constants.HUGGING_FACE_API_KEY)

    # Create dataframe
    dataframe = preprocess_dataset.create_dataframe()

    # Load documents
    loader = DataFrameLoader(dataframe, page_content_column=constants.DOCUMENT_PAGE_CONTENT_KEY)
    documents = loader.load()
    helper.print_loaded_documents(documents)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    helper.print_split_documents(chunks)
    modified_chunks = helper.modify_metadata(chunks,
                                             [constants.DOCUMENT_SHOW_ID_KEY, constants.DOCUMENT_DOCUMENT_TITLE_KEY])

    # Set up embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

    # Set up vector store
    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    helper.print_vector_store_content(vector_store)

    # Set up retriever
    retriever = vector_store.as_retriever(search_type=constants.SEARCH_TYPE,
                                          search_kwargs={constants.K_KEY: 10})

    # Set up LLM
    login(hugging_face_api_key)  # TODO verstehen, wozu das gebraucht wird
    llm = HuggingFaceEndpoint(
        repo_id=constants.LLM_MODEL_NAME,
        huggingfacehub_api_token=hugging_face_api_key,
        task=constants.LLM_TASK,
        max_new_tokens=512,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.1
    )

    return [ChatHuggingFace(llm=llm), retriever]
