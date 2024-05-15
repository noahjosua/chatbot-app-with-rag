import frontend
from initialize_app import initialize_app
import streamlit as st

if __name__ == '__main__':

    if 'chat_model' not in st.session_state:
        chat_model_and_retriever = initialize_app()
        st.session_state.chat_model = chat_model_and_retriever[0]
        st.session_state.retriever = chat_model_and_retriever[1]
    frontend.initialize_frontend()
    frontend.chat_history(st.session_state.chat_model, st.session_state.retriever)
