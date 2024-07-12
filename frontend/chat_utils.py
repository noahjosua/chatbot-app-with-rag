from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import streamlit as st

from constants import constants


def setup_qa_system(chat_model, retriever, user_prompt, prompt_template):
    rephrased_user_prompt = rephrase_user_prompt_if_necessary(chat_model, user_prompt)

    # Define prompt template for the QA system
    qa_chain_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize Language Model Chain (LLMChain)
    llm_chain = LLMChain(llm=chat_model, prompt=qa_chain_prompt, callbacks=None, verbose=True)

    # Define document prompt template
    document_prompt = PromptTemplate(
        input_variables=[constants.DOCUMENT_PAGE_CONTENT_KEY, constants.DOCUMENT_SOURCE_KEY],
        template=constants.DOCUMENT_PROMPT_TEMPLATE,
    )

    # Initialize StuffDocumentsChain to combine documents
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=constants.CONTEXT_KEY,
        document_prompt=document_prompt,
        callbacks=None,
    )

    # Initialize RetrievalQAWithSourcesChain for QA with document retrieval
    qa_chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=True,
        retriever=retriever,
        return_source_documents=True,
    )

    return [rephrased_user_prompt, qa_chain]


def rephrase_user_prompt_if_necessary(chat_model, user_prompt):
    contextualize_system_prompt_template = PromptTemplate.from_template(constants.REPHRASED_SYSTEM_PROMPT)

    # Initialize LLMChain with chat model and prompt template
    contextualize_chain = LLMChain(llm=chat_model, prompt=contextualize_system_prompt_template, callbacks=None,
                                   verbose=True)

    # Invoke chain with user prompt and return the rephrased text
    return contextualize_chain.invoke({constants.USER_PROMPT_KEY: user_prompt})[constants.TEXT_KEY]


def extract_and_format_sources(response):
    # Extract document metadata and format sources
    sources = [
        f"Entry with ID '{doc.metadata[constants.DOCUMENT_SOURCE_KEY][constants.DOCUMENT_ID_KEY]}'"
        for doc in response[constants.RESPONSE_SOURCE_DOC_KEY]]
    # Add formatted sources to response dictionary
    response[constants.RESPONSE_SOURCES_KEY] = sources
    return response


def get_chat_history():
    # Retrieve chat history from Streamlit session state and format as string
    return '\n'.join([f'{msg[constants.ROLE]}: {msg[constants.MESSAGE_CONTENT_KEY]}' for msg in
                      st.session_state.chathistory])


def format_answer_for_ui(response):
    formatted_answer = ''

    # Check if sources exist and format answer accordingly
    if len(response[constants.RESPONSE_SOURCES_KEY]) > 0:
        formatted_answer += f'{response[constants.ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITH_CONTEXT}'
        for source in response[constants.RESPONSE_SOURCES_KEY]:
            formatted_answer += f"- {source}\n\n"
    else:
        formatted_answer += f'{response[constants.ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITHOUT_CONTEXT}'
    return formatted_answer


def add_messages_to_history(user_prompt, formatted_answer, response):
    # Append user prompt and assistant response to chat history
    st.session_state.chathistory.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.chathistory.append({constants.ROLE: constants.ROLE_ASSISTANT,
                                         constants.MESSAGE_CONTENT_KEY: response[constants.ANSWER_KEY]})
    # Append user prompt and formatted answer to messages
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_ASSISTANT, constants.MESSAGE_CONTENT_KEY: formatted_answer})
