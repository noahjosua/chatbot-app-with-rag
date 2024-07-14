import streamlit as st
from constants import constants
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
            qa_system = chat_utils.setup_qa_system(chat_model, retriever, user_prompt, constants.TEMPLATE_SYSTEM_PROMPT)
            rephrased_user_prompt = qa_system[0]
            qa_chain = qa_system[1]

            # Execute QA process with user prompt and chat history
            response = qa_chain(
                {constants.QUESTION_KEY: rephrased_user_prompt, constants.CHAT_HISTORY_KEY: chat_history})

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
