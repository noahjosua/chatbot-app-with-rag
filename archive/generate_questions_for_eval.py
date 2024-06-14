import os  # Operating system functionalities
from dotenv import load_dotenv  # Environment variable handling
from huggingface_hub import login

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import pandas as pd

import constants
import helper
import preprocess_dataset

from langchain_community.document_loaders import DataFrameLoader


def setup():
    load_dotenv()
    hugging_face_api_key = os.getenv(constants.HUGGING_FACE_API_KEY)

    # Create dataframe
    dataframe = preprocess_dataset.create_dataframe(constants.DATASET_EVAL)

    # Load documents
    loader = DataFrameLoader(dataframe, page_content_column=constants.DOCUMENT_PAGE_CONTENT_KEY)
    documents = loader.load()
    modified_documents = preprocess_dataset.strip_unnecessary_prefixes_from_metadata(documents)
    helper.print_loaded_documents(modified_documents)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    helper.print_split_documents(chunks)
    modified_chunks = helper.modify_metadata(chunks,
                                             [constants.DOCUMENT_SHOW_ID_KEY, "title",
                                              constants.DOCUMENT_DOCUMENT_TITLE_KEY])
    # Set up LLM
    login(hugging_face_api_key)
    question_generation_pipeline = pipeline('text2text-generation', model='google/flan-t5-base')

    questions = []
    for chunk in modified_chunks:
        print(chunk)
        # Extract metadata
        show_id = chunk.metadata[constants.DOCUMENT_SOURCE_KEY][constants.DOCUMENT_SHOW_ID_KEY]
        show_title = chunk.metadata[constants.DOCUMENT_SOURCE_KEY]['title']
        chunk_content = chunk.page_content
        print(show_id)
        print(show_title)
        print(chunk_content)

        # Prepare the prompt for question generation with metadata reference
        prompt = f"Generate a question about the following text (Show ID: {show_id}, Title: {show_title}): {chunk_content}"

        # Generate the question
        result = question_generation_pipeline(prompt, max_length=100)
        print(result)

        # Extract the generated question and append metadata reference
        question = result[0]['generated_text']
        question_with_reference = f"{question} (Reference: Show ID = {show_id}, Title = {show_title})"
        questions.append(question_with_reference)

    for question in questions:
        print(question)

    # Save the questions to a new CSV file
    questions_df = pd.DataFrame({'generated_question': questions})
    questions_df.to_csv('generated_questions.csv', index=True)
    print("Questions generated and saved to generated_questions.csv")
