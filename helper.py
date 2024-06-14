from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

import constants


def modify_metadata(chunks): #, keys_to_keep):
    modified_chunks = []

    for chunk in chunks:
        # original_metadata = chunk.metadata

        # Retain only the specified keys in the metadata
        # modified_metadata = {key: original_metadata[key] for key in keys_to_keep if key in original_metadata}

        # Create a new dictionary with the 'source' key holding the modified_metadata
        new_metadata = {constants.DOCUMENT_SOURCE_KEY: chunk.metadata}

        # Update the document with the modified metadata
        chunk.metadata = new_metadata
        modified_chunks.append(chunk)
        print(f"CHUNK: {chunk}")
    return modified_chunks


def rephrase_user_prompt_if_necessary(chat_model, chat_history_text, user_prompt):
    # load contextualized system prompt
    contextualize_system_prompt = constants.CONTEXTUALIZED_SYSTEM_PROMPT

    contextualize_system_prompt_template = PromptTemplate.from_template(contextualize_system_prompt)
    contextualize_chain = LLMChain(llm=chat_model, prompt=contextualize_system_prompt_template, callbacks=None,
                                   verbose=True)

    # rephrase user prompt if necessary
    return contextualize_chain.invoke(
        {constants.CHAT_HISTORY_KEY: chat_history_text, constants.USER_PROMPT_KEY: user_prompt})[
        constants.TEXT_KEY]


def handle_sources(response):
    # Extract and format sources from source_documents
    sources = [
        f"Entry with '{doc.metadata[constants.DOCUMENT_SOURCE_KEY][constants.DOCUMENT_ID_KEY]}'"
        for doc in response[constants.RESPONSE_SOURCE_DOC_KEY]]

    # Update the response with formatted sources
    response[constants.RESPONSE_SOURCES_KEY] = sources
    return response


def format_answer_for_ui(response):
    formatted_answer = ''
    if len(response[constants.RESPONSE_SOURCES_KEY]) > 0:
        formatted_answer += f'{response[constants.RESPONSE_ANSWER_KEY]}{constants.FORMATTED_ANSWER_WITH_CONTEXT}'
        for source in response[constants.RESPONSE_SOURCES_KEY]:
            formatted_answer += f"- {source}\n\n"
    else:
        formatted_answer += f'{response[constants.RESPONSE_SOURCES_KEY]}{constants.FORMATTED_ANSWER_WITHOUT_CONTEXT}'
    return formatted_answer


def print_vector_store_content(vector_store):
    # Access the stored embeddings and documents
    db_index = vector_store.index
    stored_docs = vector_store.docstore._dict

    # Create a mapping of document IDs to their integer indices
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(stored_docs.keys())}

    # Iterate through the documents and embeddings
    for doc_id, document in stored_docs.items():
        index = doc_id_to_index[doc_id]
        embedding = db_index.reconstruct(index)

        print(f'Document {doc_id}: {document}\n')
        print(f'Embedding {doc_id}: {embedding}\n')


def print_loaded_documents(documents):
    print('loaded documents...')
    for document in documents:
        print(document)


def print_split_documents(documents):
    print('split documents...')
    i = 0
    while i < len(documents):
        current_text = documents[i]
        print(current_text)
        i += 1
