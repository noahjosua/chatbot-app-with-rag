import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate

import helper


def initialize_frontend():
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header('My Chatbot')
    with st.chat_message('assistant'):
        st.write('Hello 👋')

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


def chat_history(chat_model, retriever):
    # React to user input
    if user_prompt := st.chat_input('Type your question'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(user_prompt)

        chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

        for msg in st.session_state.chat_history:
            print(msg)

        with (st.spinner('Thinking...')):

            template = """
            Use the following pieces of retrieved context as well as the chat history to answer the question 
            (which might reference context in the chat history). 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Keep the answer as concise as possible. 
            Context:\n {context}
            Chat History:\n {chat_history}
            Question:\n {question}
            """

            '''
            system_prompt_template = """
            Use the following pieces of retrieved context as well as the chat history to answer the question 
            (which might reference context in the chat history). 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Keep the answer as concise as possible. 
            Context: {context}
            Chat History: {chat_history}
            """

            human_prompt_template = "[INST]{question}[/INST]"

            qa_chain_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', system_prompt_template),
                    ('human', human_prompt_template),
                ]
            )
            '''
            qa_chain_prompt = PromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=chat_model, prompt=qa_chain_prompt, callbacks=None, verbose=True)

            document_prompt = PromptTemplate(
                input_variables=['page_content', 'source'],
                template='Content:\n{page_content}\nSource:{source}',
            )

            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name='context',
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

            response = qa({'question': user_prompt, 'chat_history': chat_history_text})
            print(response)

            # Extract and format sources from source_documents
            sources = [
                f"Document with title {doc.metadata['source'].replace('document: ', '')}, Entry with show_id: {doc.metadata['show_id']}"
                for doc in response['source_documents']]

            # Update the response with formatted sources
            response['sources'] = sources

            formatted_answer = f"{response['answer']}\n\n**Used context to answer your question:**\n\n"
            for source in response['sources']:
                formatted_answer += f"- {source}\n\n"

        # Display assistant response in chat message container
        with st.chat_message('assistant'):
            st.markdown(formatted_answer)

        # Add user and assistant messages to chat history
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})
        st.session_state.messages.append({'role': 'assistant', 'content': formatted_answer})
        st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})
        st.session_state.chat_history.append({'role': 'assistant', 'content': response['answer']})

    else:
        pass
