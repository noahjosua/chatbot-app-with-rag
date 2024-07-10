import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate

import constants
from frontend import chat_utils


def chat_flow(chat_model, retriever):
    # Check if user input is provided
    if user_prompt := st.chat_input(constants.CHAT_INPUT):
        # Display user prompt
        with st.chat_message(constants.ROLE_USER):
            st.markdown(user_prompt)

        # Use a spinner to indicate processing
        with ((st.spinner(constants.SPINNER_LABEL))):
            chat_history = chat_utils.get_chat_history()
            rephrased_user_prompt = chat_utils.rephrase_user_prompt_if_necessary(chat_model, user_prompt)

            # Define prompt template for the QA system
            template_system_prompt = constants.TEMPLATE_SYSTEM_PROMPT
            qa_chain_prompt = ChatPromptTemplate.from_template(template_system_prompt)

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
                document_variable_name=constants.DOCUMENT_PROMPT_VARIABLE_NAME,
                document_prompt=document_prompt,
                callbacks=None,
            )

            # Initialize RetrievalQAWithSourcesChain for QA with document retrieval
            qa = RetrievalQAWithSourcesChain(
                combine_documents_chain=combine_documents_chain,
                callbacks=None,
                verbose=True,
                retriever=retriever,
                return_source_documents=True,
            )

            # Execute QA process with user prompt and chat history
            response = qa(
                {constants.QA_USER_PROMPT_KEY: rephrased_user_prompt, constants.CHAT_HISTORY_KEY: chat_history})

            # Extract and format sources from the response
            updated_response = chat_utils.extract_and_format_sources(response)
            formatted_answer = chat_utils.format_answer_for_ui(updated_response)

        # Display assistant's response
        with st.chat_message(constants.ROLE_ASSISTANT):
            st.markdown(formatted_answer)

        # Add user prompt, formatted answer, and response to chat history
        chat_utils.add_messages_to_history(user_prompt, formatted_answer, response)
        # print_to_console.print_chat_history()
    else:
        pass
