import os  # Operating system functionalities
from dotenv import load_dotenv  # Environment variable handling
from huggingface_hub import login
from langchain_community.chat_models import ChatHuggingFace

# Langchain #
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import preprocess_dataset
import constants
import helper


def initial_setup():
    load_dotenv()
    hugging_face_api_key = os.getenv(constants.HUGGING_FACE_API_KEY)

    # Create dataframe
    dataframe = preprocess_dataset.create_dataframe(constants.DATASET_EVAL) # TODO umstellen auf das große set

    # Replace NaN values with "unknown"
    dataframe.fillna(constants.REPLACEMENT_NAN_VALUES, inplace=True)

    # Load documents
    loader = DataFrameLoader(dataframe, page_content_column=constants.DOCUMENT_PAGE_CONTENT_KEY)
    documents = loader.load()
    modified_documents = preprocess_dataset.strip_unnecessary_prefixes_from_metadata(documents)
    helper.print_loaded_documents(modified_documents)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    helper.print_split_documents(chunks)
    modified_chunks = helper.modify_metadata(chunks) #,[constants.DOCUMENT_SHOW_ID_KEY, constants.DOCUMENT_DOCUMENT_TITLE_KEY])

    # Set up embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

    # Set up vector store
    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    helper.print_vector_store_content(vector_store)

    # Set up retriever
    retriever = vector_store.as_retriever(search_type=constants.SEARCH_TYPE,
                                          search_kwargs={constants.K_KEY: 5})

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
