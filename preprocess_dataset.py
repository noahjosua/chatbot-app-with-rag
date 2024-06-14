import pandas as pd
from langchain_core import documents

import constants


def create_dataframe(name_of_dataset):
    dataframe = pd.read_csv(name_of_dataset)

    # Replace NaN values with "unknown"
    dataframe.fillna(constants.REPLACEMENT_NAN_VALUES, inplace=True)

    dataframe = _add_column_name_to_row_value(dataframe)

    # Combine all columns into a single column, removing extra spaces
    dataframe[constants.DOCUMENT_PAGE_CONTENT_KEY] = dataframe.apply(lambda row: '; '.join(row.astype(str).str.strip()),
                                                                     axis=1)
    # TODO NOAH auslagern
    pd.set_option('display.max_colwidth', None)
    for index, row in dataframe.iterrows():
        print(row[constants.DOCUMENT_PAGE_CONTENT_KEY])

    return dataframe


def _add_column_name_to_row_value(dataframe):
    # Iterate through each row
    for index, row in dataframe.iterrows():
        # Iterate through each column
        for column in dataframe.columns:
            # Convert the field value to string and concatenate the column name
            dataframe.at[index, column] = f"{column}: {str(row[column])}"
    return dataframe


def strip_unnecessary_prefixes_from_metadata(loaded_documents):
    for document in loaded_documents:
        processed_metadata = {key.split(': ')[0]: value.split(': ')[1] for key, value in document.metadata.items()}
        document.metadata = processed_metadata

        # Print the page_content and processed metadata
        print(f"page_content='{document.page_content}' metadata={document.metadata}")
    return loaded_documents
