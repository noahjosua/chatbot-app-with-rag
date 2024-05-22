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

        contextualize_system_prompt = """Create a self-contained question from a given user prompt, 
        considering possible references to previous chat history. 
        If the user's prompt is unrelated to the chat history, leave it unchanged.
        If the user's prompt references the chat history, rephrase the question to make it comprehensible without 
        the need for the context provided by the chat history.
        Please refrain from answering the question; simply adjust its wording if necessary, otherwise leave it unchanged.
        Chat History: {chat_history}
        User Prompt: {user_prompt}"""

        contextualize_system_prompt_template = PromptTemplate.from_template(contextualize_system_prompt)
        contextualize_chain = LLMChain(llm=chat_model, prompt=contextualize_system_prompt_template, callbacks=None,
                                       verbose=True)

        rephrased_user_prompt = \
            contextualize_chain.invoke({'chat_history': chat_history_text, 'user_prompt': user_prompt})['text']
        print(rephrased_user_prompt)

        with (st.spinner('Thinking...')):

            template = """
            User: {question}
            Assistant: Use the following pieces of retrieved context as well as the chat history to answer the question 
            (which might reference context in the chat history). If you don't know the answer, 
            just say that you don't know, don't try to make up an answer. 
            If you don't get context, don't refer to it in your answer. Keep the answer as concise as possible.
            Context:\n {context}
            Chat History:\n {chat_history}
            """

            qa_chain_prompt = ChatPromptTemplate.from_template(template)

            llm_chain = LLMChain(llm=chat_model, prompt=qa_chain_prompt, callbacks=None, verbose=True)

            document_prompt = PromptTemplate(
                input_variables=['page_content', 'source'],
                template='Content:\n{page_content}\nMetadata:{source}',
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

            response = qa({'question': rephrased_user_prompt, 'chat_history': chat_history_text})

            # Extract and format sources from source_documents
            sources = [
                f"Document with title '{doc.metadata['source']['document_title']}', Entry with show_id '{doc.metadata['source']['show_id']}'"
                for doc in response['source_documents']]

            # Update the response with formatted sources
            response['sources'] = sources

            if len(sources) > 0:
                formatted_answer = f"{response['answer']}\n\n**Used context to answer your question:**\n\n"
                for source in sources:
                    formatted_answer += f"- {source}\n\n"
            else:
                formatted_answer = f"{response['answer']}\n\nThere are no documents that could be used to answer your question."

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
