import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate

import constants
import helper


def chat_flow(chat_model, retriever):
    if user_prompt := st.chat_input(constants.CHAT_INPUT):

        # Display user message in chat message container
        with st.chat_message(constants.ROLE_USER):
            st.markdown(user_prompt)

        with (st.spinner(constants.SPINNER_LABEL)):

            # construct chat history
            chat_history_text = '\n'.join(
                [f'{msg[constants.ROLE]}: {msg[constants.MESSAGE_CONTENT_KEY]}' for msg in st.session_state.chat_flow])

            # rephrase user prompt if necessary
            rephrased_user_prompt = helper.rephrase_user_prompt_if_necessary(chat_model, chat_history_text, user_prompt)

            # load template for system prompt
            template_system_prompt = constants.TEMPLATE_SYSTEM_PROMPT
            qa_chain_prompt = ChatPromptTemplate.from_template(template_system_prompt)
            llm_chain = LLMChain(llm=chat_model, prompt=qa_chain_prompt, callbacks=None, verbose=True)

            # construct document prompt and chains
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

            # invoke RetrievalQAWithSourcesChain with rephrased user prompt and chat history
            response = qa(
                {constants.QA_USER_PROMPT_KEY: rephrased_user_prompt, constants.CHAT_HISTORY_KEY: chat_history_text})

            # handle sources
            updated_response = helper.handle_sources(response)

            # format answer
            formatted_answer = helper.format_answer_for_ui(updated_response)

        # Display assistant response in chat message container
        with st.chat_message(constants.ROLE_ASSISTANT):
            st.markdown(formatted_answer)

        # Add user and assistant messages to chat history
        st.session_state.messages.append(
            {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
        st.session_state.messages.append(
            {constants.ROLE: constants.ROLE_ASSISTANT, constants.MESSAGE_CONTENT_KEY: formatted_answer})
        st.session_state.chat_flow.append(
            {constants.ROLE: constants.ROLE_USER, constants.MESSAGE_CONTENT_KEY: user_prompt})
        st.session_state.chat_flow.append({constants.ROLE: constants.ROLE_ASSISTANT,
                                           constants.MESSAGE_CONTENT_KEY: response[constants.RESPONSE_ANSWER_KEY]})
    else:
        pass

# 2, 3, 4, 5, 6, 9 How many entries are of type TV Show?
