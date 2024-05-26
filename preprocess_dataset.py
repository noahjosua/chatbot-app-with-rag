import pandas as pd

import constants


def create_dataframe():
    dataframe = pd.read_csv(constants.DATASET)

    # Replace NaN values with "unknown"
    dataframe.fillna(constants.REPLACEMENT_NAN_VALUES, inplace=True)

    dataframe = _add_column_name_to_row_value(dataframe)

    # Combine all columns into a single column, removing extra spaces
    dataframe[constants.DOCUMENT_PAGE_CONTENT_KEY] = dataframe.apply(lambda row: '; '.join(row.astype(str).str.strip()),
                                                                     axis=1)

    return dataframe


def _add_column_name_to_row_value(dataframe):
    # Iterate through each row
    for index, row in dataframe.iterrows():
        # Iterate through each column
        for column in dataframe.columns:
            # Convert the field value to string and concatenate the column name
            dataframe.at[index, column] = f"{column}: {str(row[column])}"
    return dataframe
