import os
import pandas as pd
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import constants
from frontend import chat_utils


def evaluation(chat_model, retriever):
    dataframe = prepare_dataframe()
    questions = collect_questions(dataframe)
    answers = collect_answers(dataframe)

    question = questions[9][1] # TODO rows löschen, wo null werte oder komische zeichenketten und nichts sinnvolles drin sind
    expected_answer = answers[9][1]
    print(question)
    print(expected_answer)

    actual_answer_and_context_answer = flow(chat_model, retriever, question)
    actual_answer = actual_answer_and_context_answer[0]
    context_answer = actual_answer_and_context_answer[1]
    print(actual_answer)
    print(context_answer)

    '''
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=[
            context_answer
        ]
    )

    os.getenv("OPENAI_API_KEY")
    contextual_precision_metric = ContextualPrecisionMetric()
    contextual_recall_metric = ContextualRecallMetric()
    contextual_relevancy_metric = ContextualRelevancyMetric()

    contextual_precision_metric.measure(test_case)
    print("Score: ", contextual_precision_metric.score)
    print("Reason: ", contextual_precision_metric.reason)

    contextual_recall_metric.measure(test_case)
    print("Score: ", contextual_recall_metric.score)
    print("Reason: ", contextual_recall_metric.reason)

    contextual_relevancy_metric.measure(test_case)
    print("Score: ", contextual_relevancy_metric.score)
    print("Reason: ", contextual_relevancy_metric.reason)
    '''


def prepare_dataframe():
    dataframe = pd.read_csv(constants.DATASET_EVAL)
    dataframe.fillna(constants.REPLACEMENT_NAN_VALUES, inplace=True)
    return dataframe


def collect_questions(dataframe):
    questions = []
    for index, row in dataframe.iterrows():
        questions.append((row['id'], row['question']))
    return questions


def collect_answers(dataframe):
    answers = []
    for index, row in dataframe.iterrows():
        answers.append((row['id'], row['answer']))
    return answers


TEMPLATE_SYSTEM_PROMPT = """
User: {question}
Assistant: To answer the question, you should follow these steps:

1. Check if the provided context is not empty.
2. Important: If context is empty, respond with "I don't know the answer to that question based on the provided information."
3. If context is not empty:
    a. Review the provided context carefully to see if it contains relevant information to answer the question.
    b. If the context contains enough information to answer the question, provide a clear and concise answer based on that information.
    c. If the context does not contain enough information to answer the question, respond with "I don't know the answer to that question based on the provided information."
4. Do not make up answers or provide speculative information if the context does not contain relevant information to answer the question.\n

Here is the context:\n {context}
"""


def flow(chat_model, retriever, question):
    rephrased_user_prompt = chat_utils.rephrase_user_prompt_if_necessary(chat_model, question)

    qa_chain_prompt = ChatPromptTemplate.from_template(TEMPLATE_SYSTEM_PROMPT)
    llm_chain = LLMChain(llm=chat_model, prompt=qa_chain_prompt, callbacks=None, verbose=True)

    document_prompt = PromptTemplate(
        input_variables=[constants.DOCUMENT_PAGE_CONTENT_KEY, constants.DOCUMENT_SOURCE_KEY],
        template=constants.DOCUMENT_PROMPT_TEMPLATE,
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=constants.DOCUMENT_PROMPT_VARIABLE_NAME,
        document_prompt=document_prompt,
        callbacks=None,
    )
    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=True,
        retriever=retriever,
        return_source_documents=True,
    )

    response = qa(
        {constants.QA_USER_PROMPT_KEY: rephrased_user_prompt})
    print(response)
    actual_answer = response['answer']
    source_documents = response['source_documents']
    context_answer = ''
    for document in source_documents:
        context_answer = document.metadata['source']['answer']
        # TODO was tun, wenn es mehrere documente gibt? oder wenn es keine gibt?
    return [actual_answer, context_answer]
