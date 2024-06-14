import streamlit as st

import constants


def initialize_frontend():
    # Initialize chat history
    if constants.MESSAGES_KEY not in st.session_state:
        st.session_state.messages = []
    if constants.CHAT_HISTORY_KEY not in st.session_state:
        st.session_state.chat_flow = []

    st.header(constants.CHAT_HEADER)
    with st.chat_message(constants.ROLE_ASSISTANT):
        st.write(constants.CHAT_GREETING)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message[constants.ROLE]):
            st.markdown(message[constants.MESSAGE_CONTENT_KEY])
