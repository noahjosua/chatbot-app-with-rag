from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

import streamlit as st

import constants


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
        formatted_answer += f'{response[constants.RESPONSE_ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITH_CONTEXT}'
        for source in response[constants.RESPONSE_SOURCES_KEY]:
            formatted_answer += f"- {source}\n\n"
    else:
        formatted_answer += f'{response[constants.RESPONSE_ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITHOUT_CONTEXT}'
    return formatted_answer


def add_messages_to_history(user_prompt, formatted_answer, response):
    # Append user prompt and assistant response to chat history
    st.session_state.chathistory.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.chathistory.append({constants.ROLE: constants.ROLE_ASSISTANT,
                                         constants.MESSAGE_CONTENT_KEY: response[constants.RESPONSE_ANSWER_KEY]})
    # Append user prompt and formatted answer to messages
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_ASSISTANT, constants.MESSAGE_CONTENT_KEY: formatted_answer})
