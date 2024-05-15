import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import helper


def initialize_frontend():
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.header('My Chatbot')
    with st.chat_message('assistant'):
        st.write('Hello 👋')

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


def chat_history(chat_model, retriever):
    # React to user input
    if prompt := st.chat_input('Type your question'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.spinner('Thinking...'):
            response = chat_model.invoke(prompt)

            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer as concise as possible. 
            {context}
            Question: {question}
            Helpful Answer:"""
            qa_chain_prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

            docs = retriever.invoke(prompt)
            helper.print_docs_for_question(docs)

            # set up chain
            chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | qa_chain_prompt
                    | chat_model
                    | StrOutputParser()
            )

            response = chain.invoke(prompt)
            print(f'Result: {response}')

            # Display assistant response in chat message container
            with st.chat_message('assistant'):
                st.markdown(response)

            # Add user and assistant messages to chat history
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.session_state.messages.append({'role': 'assistant', 'content': response})
