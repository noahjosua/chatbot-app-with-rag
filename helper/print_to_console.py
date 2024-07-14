from constants import constants

import streamlit as st


def print_dataframe(dataframe):
    print('Dataframe...\n')
    for index, row in dataframe.iterrows():
        print(row[constants.DOCUMENT_PAGE_CONTENT_KEY])
    print('\n')


def print_loaded_documents(documents):
    print('Loaded documents...\n')
    for document in documents:
        print(document)
    print('\n')


def print_split_documents(documents):
    print('Split documents...\n')
    i = 0
    while i < len(documents):
        current_text = documents[i]
        print(current_text)
        i += 1
    print('\n')


def print_documents_with_modified_metadata(chunks):
    print('Documents with modified metadata...\n')
    for chunk in chunks:
        print(chunk)
    print('\n')


def print_vector_store_content(vector_store):
    print('Vector store...\n')
    # Access the stored embeddings and documents
    db_index = vector_store.index
    stored_docs = vector_store.docstore._dict

    # Create a mapping of document IDs to their integer indices
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(stored_docs.keys())}

    # Iterate through the documents and embeddings
    for doc_id, document in stored_docs.items():
        index = doc_id_to_index[doc_id]
        embedding = db_index.reconstruct(index)

        print(f'Document {doc_id}: {document}')
        print(f'Embedding {doc_id}: {embedding}\n')
    print('\n')


def print_chat_history():
    print('Chat history...\n')
    for message in st.session_state.chathistory:
        print(message)
    print('\n')

    print('Messages...\n')
    for message in st.session_state.messages:
        print(message)
    print('\n')


def print_modified_metadata(documents):
    print('Strip unnecessary prefixes from metadata...\n')
    for document in documents:
        print(f"page_content='{document.page_content}' metadata={document.metadata}")
    print('\n')
