import os  # Operating system functionalities
from dotenv import load_dotenv  # Environment variable handling

import pandas as pd  # Todo

# Langchain #
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

from huggingface_hub import login
import helper


def initialize_app():
    load_dotenv()
    hugging_face_api_key = os.getenv('HUGGING_FACE_API_KEY')

    # Todo
    dataframe = pd.read_csv('netflix_titles.csv')

    # Set the option to display all columns and the full content
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)

    # Replace NaN values with "unknown"
    dataframe.fillna('unknown', inplace=True)

    # Create a separate column 'document_title' and fill it with the title
    dataframe['document_title'] = 'netflix_titles.csv'

    # Combine all columns into a single column, removing extra spaces
    dataframe['page_content'] = dataframe.apply(lambda row: '; '.join(row.astype(str).str.strip()), axis=1)

    # Load documents
    loader = DataFrameLoader(dataframe, page_content_column='page_content')
    documents = loader.load()
    # helper.print_loaded_documents(documents)

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    # print(helper.print_split_documents(chunks))
    modified_chunks = helper.modify_metadata(chunks, ['show_id', 'document_title'])

    # set up embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    # set up vector db
    vector_store = FAISS.from_documents(modified_chunks, embeddings)
    # helper.print_vector_store_content(vector_store)

    # set up retriever
    retriever = vector_store.as_retriever(search_type="similarity",
                                          search_kwargs={"k": 10})

    # set up LLM
    login(hugging_face_api_key)  # TODO verstehen, wozu das gebraucht wird
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
