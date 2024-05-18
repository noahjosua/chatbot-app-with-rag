import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
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
    if user_prompt := st.chat_input('Type your question'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(user_prompt)

        with (st.spinner('Thinking...')):
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Keep the answer as concise as possible. 
            {context}
            Question: {question}"""

            qa_chain_prompt = PromptTemplate.from_template(prompt_template)

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

            response = qa(user_prompt)

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

    else:
        pass
