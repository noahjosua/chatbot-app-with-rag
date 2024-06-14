import pandas as pd


def generate_answers_to_questions():
    dataframe = pd.read_csv("generated_questions.csv")

    pd.set_option('display.max_colwidth', None)
    for index, row in dataframe.iterrows():
        question = row['generated_question']
        print(f"{index}, {question}")