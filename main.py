import constants
from frontend.initialize import initialize_frontend
from frontend.chat_flow import chat_flow
from initial_setup import initial_setup
import streamlit as st

if __name__ == '__main__':

    if constants.CHAT_MODEL_IN_SESSION_STATE not in st.session_state:
        chat_model_and_retriever = initial_setup()
        st.session_state.chat_model = chat_model_and_retriever[0]
        st.session_state.retriever = chat_model_and_retriever[1]
    initialize_frontend()
    chat_flow(st.session_state.chat_model, st.session_state.retriever)
