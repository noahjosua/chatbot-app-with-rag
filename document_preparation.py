from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

import constants
from helper import print_to_console


def setup_documents():
    dataframe = _create_dataframe(constants.DATASET_EVAL)

    loader = DataFrameLoader(dataframe, page_content_column=constants.DOCUMENT_PAGE_CONTENT_KEY)
    documents = loader.load()
    modified_documents = _strip_unnecessary_prefixes_from_metadata(documents)
    print_to_console.print_loaded_documents(modified_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    print_to_console.print_split_documents(chunks)
    return _modify_metadata(chunks)


def _create_dataframe(name_of_dataset):
    dataframe = pd.read_csv(name_of_dataset)
    dataframe.fillna(constants.REPLACEMENT_NAN_VALUES, inplace=True)
    dataframe = _add_column_name_to_row_value(dataframe)

    # Combine all columns into a single column, removing extra spaces
    dataframe[constants.DOCUMENT_PAGE_CONTENT_KEY] = dataframe.apply(lambda r: '; '.join(r.astype(str).str.strip()),
                                                                     axis=1)
    pd.set_option('display.max_colwidth', None)
    print_to_console.print_dataframe(dataframe)
    return dataframe


def _add_column_name_to_row_value(dataframe):
    for index, row in dataframe.iterrows():
        for column in dataframe.columns:
            # Convert the field value to string and concatenate the column name
            dataframe.at[index, column] = f"{column}: {str(row[column])}"
    return dataframe


def _strip_unnecessary_prefixes_from_metadata(loaded_documents):
    for document in loaded_documents:
        processed_metadata = {key.split(': ')[0]: value.split(': ')[1] for key, value in document.metadata.items()}
        document.metadata = processed_metadata
    print_to_console.print_documents_with_modified_metadata(loaded_documents)
    return loaded_documents


def _modify_metadata(chunks):
    modified_chunks = []
    for chunk in chunks:
        # Create a new dictionary with the 'source' key holding the modified_metadata
        new_metadata = {constants.DOCUMENT_SOURCE_KEY: chunk.metadata}
        chunk.metadata = new_metadata
        modified_chunks.append(chunk)
    return modified_chunks
