import json
import os
import pandas as pd
from constants import constants
from frontend import chat_utils


def generate_json_for_eval(chat_model, retriever, dataset, json_file_name):
    dataframe = _prepare_dataframe(dataset)
    questions_for_eval = _collect_questions(dataframe)
    answers_for_eval = _collect_answers(dataframe)
    _generate_json(questions_for_eval, answers_for_eval, chat_model, retriever, json_file_name)


def _prepare_dataframe(dataset):
    dataframe = pd.read_csv(dataset)
    mask = dataframe[constants.ANSWER_KEY].str.contains(constants.UNUSABLE_ROW_KEY)
    filtered_dataframe = dataframe[~mask]
    return filtered_dataframe


def _collect_questions(dataframe):
    questions = []
    for index, row in dataframe.iterrows():
        questions.append((row[constants.DOCUMENT_ID_KEY], row[constants.QUESTION_KEY]))
    return questions


def _collect_answers(dataframe):
    answers = []
    for index, row in dataframe.iterrows():
        answers.append((row[constants.DOCUMENT_ID_KEY], row[constants.ANSWER_KEY]))
    return answers


def _generate_json(questions_for_eval, answers_for_eval, chat_model, retriever, json_file_name):
    results = []
    for question_id, question in questions_for_eval:
        expected_answer = next(answer for answer_id, answer in answers_for_eval if answer_id == question_id)
        actual_answer_and_context_as_list = _chat_flow(chat_model, retriever, question)
        actual_answer = actual_answer_and_context_as_list[0]
        context_as_list = actual_answer_and_context_as_list[1]
        context_string = ' '.join(context_as_list)

        results.append({
            constants.QUESTION_ID_KEY: question_id,
            constants.QUESTION_KEY: question,
            constants.EXPECTED_ANSWER_KEY: expected_answer,
            constants.ACTUAL_ANSWER_KEY: actual_answer,
            constants.CONTEXT_KEY: context_string
        })

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the JSON file
    json_path = os.path.join(current_dir, json_file_name)

    # Save the results to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def _chat_flow(chat_model, retriever, question):
    qa_system = chat_utils.setup_qa_system(chat_model, retriever, question, constants.TEMPLATE_SYSTEM_PROMPT_FOR_EVAL)
    rephrased_user_prompt = qa_system[0]
    qa_chain = qa_system[1]

    response = qa_chain(
        {constants.QUESTION_KEY: rephrased_user_prompt})
    print(response)

    actual_answer = response[constants.ANSWER_KEY]
    source_documents = response[constants.RESPONSE_SOURCE_DOC_KEY]

    context_as_list = []
    for document in source_documents:
        context_as_list.append(document.metadata[constants.DOCUMENT_SOURCE_KEY][constants.ANSWER_KEY])

    return [actual_answer, context_as_list]
