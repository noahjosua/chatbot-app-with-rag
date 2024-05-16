import streamlit as st
from langchain.prompts import PromptTemplate
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
    if prompt := st.chat_input('Type your question'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)

        with (st.spinner('Thinking...')):
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Keep the answer as concise as possible. 
            {context}
            Question: {question}
            Helpful Answer:"""

            prompt_template = PromptTemplate(input_variables=['context', 'question'], template=template)

            # set up chain
            chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | chat_model
                    | StrOutputParser()
            )

        response = chain.invoke(prompt)
        print(f'Result: {response}')

        '''
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=chat_model,
                                                            chain_type='stuff',
                                                            retriever=retriever,
                                                            return_source_documents=True,
                                                            verbose=True
                                                            )

        # run chain
        response = chain({'question': prompt})
        '''

        # Display assistant response in chat message container
        with st.chat_message('assistant'):
            st.markdown(response)

        # Add user and assistant messages to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.messages.append({'role': 'assistant', 'content': response})
