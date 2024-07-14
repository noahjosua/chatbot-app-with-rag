from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

from constants import constants


def setup_documents(dataset):
    dataframe = _create_dataframe(dataset)
    loader = DataFrameLoader(dataframe, page_content_column=constants.DOCUMENT_PAGE_CONTENT_KEY)
    documents = loader.load()
    modified_documents = _strip_unnecessary_prefixes_from_metadata(documents)
    # print_to_console.print_loaded_documents(modified_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(modified_documents)
    # print_to_console.print_split_documents(chunks)
    return _modify_metadata(chunks)


def _create_dataframe(dataset):
    dataframe = pd.read_csv(dataset)
    mask = dataframe[constants.ANSWER_KEY].str.contains(constants.UNUSABLE_ROW_KEY)
    filtered_dataframe = dataframe[~mask]
    new_dataframe = _add_column_name_to_row_value(filtered_dataframe)

    # Combine all columns into a single column, removing extra spaces
    new_dataframe[constants.DOCUMENT_PAGE_CONTENT_KEY] = new_dataframe.apply(
        lambda r: '; '.join(r.astype(str).str.strip()),
        axis=1)
    # pd.set_option('display.max_colwidth', None)  # Set option to display full column width
    # print_to_console.print_dataframe(new_dataframe)
    return new_dataframe


def _add_column_name_to_row_value(dataframe):
    dataframe = dataframe.astype(str)
    for index, row in dataframe.iterrows():
        for column in dataframe.columns:
            # Convert the field value to string and concatenate the column name
            dataframe.at[index, column] = f"{column}: {str(row[column])}"
    return dataframe


def _strip_unnecessary_prefixes_from_metadata(loaded_documents):
    for document in loaded_documents:
        # Process metadata by splitting keys and values and updating document metadata
        processed_metadata = {key.split(': ')[0]: value.split(': ')[1] for key, value in document.metadata.items()}
        document.metadata = processed_metadata
    # print_to_console.print_documents_with_modified_metadata(loaded_documents)
    return loaded_documents


def _modify_metadata(chunks):
    modified_chunks = []
    for chunk in chunks:
        # Create a new dictionary with the 'source' key holding the modified_metadata
        new_metadata = {constants.DOCUMENT_SOURCE_KEY: chunk.metadata}
        chunk.metadata = new_metadata
        modified_chunks.append(chunk)
    return modified_chunks
