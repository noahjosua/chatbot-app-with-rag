from constants import constants
from init import initialization
from frontend.chat_flow import chat_flow
from frontend.frontend_initialization import initialize_frontend
import streamlit as st

if __name__ == '__main__':

    if constants.CHAT_MODEL_IN_SESSION_STATE not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            placeholder.write('Hang in there - Everything is being set up for you...')

        chat_model_and_retriever = initialization.initial_setup(constants.DATASET_SHORTENED) # noch ändern
        st.session_state.chat_model = chat_model_and_retriever[0]
        st.session_state.retriever = chat_model_and_retriever[1]
        placeholder.empty()

    initialize_frontend()
    chat_flow(st.session_state.chat_model, st.session_state.retriever)
