from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

import streamlit as st

import constants


def rephrase_user_prompt_if_necessary(chat_model, chat_history_text, user_prompt):
    contextualize_system_prompt = constants.CONTEXTUALIZED_SYSTEM_PROMPT
    contextualize_system_prompt_template = PromptTemplate.from_template(contextualize_system_prompt)
    contextualize_chain = LLMChain(llm=chat_model, prompt=contextualize_system_prompt_template, callbacks=None,
                                   verbose=True)
    return contextualize_chain.invoke(
        {constants.CHAT_HISTORY_KEY: chat_history_text, constants.USER_PROMPT_KEY: user_prompt})[
        constants.TEXT_KEY]


def extract_and_format_sources(response):
    sources = [
        f"Entry with ID '{doc.metadata[constants.DOCUMENT_SOURCE_KEY][constants.DOCUMENT_ID_KEY]}'"
        for doc in response[constants.RESPONSE_SOURCE_DOC_KEY]]

    response[constants.RESPONSE_SOURCES_KEY] = sources
    return response


def format_answer_for_ui(response):
    formatted_answer = ''
    if len(response[constants.RESPONSE_SOURCES_KEY]) > 0:
        formatted_answer += f'{response[constants.RESPONSE_ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITH_CONTEXT}'
        for source in response[constants.RESPONSE_SOURCES_KEY]:
            formatted_answer += f"- {source}\n\n"
    else:
        formatted_answer += f'{response[constants.RESPONSE_SOURCES_KEY]}{constants.FORMATTED_ANSWER_WITHOUT_CONTEXT}'
    return formatted_answer


def add_messages_to_history(user_prompt, formatted_answer, response):
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.messages.append(
        {constants.ROLE: constants.ROLE_ASSISTANT, constants.MESSAGE_CONTENT_KEY: formatted_answer})
    st.session_state.chat_flow.append(
        {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
    st.session_state.chat_flow.append({constants.ROLE: constants.ROLE_ASSISTANT,
                                       constants.MESSAGE_CONTENT_KEY: response[constants.RESPONSE_ANSWER_KEY]})