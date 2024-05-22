import os


def modify_metadata(chunks, keys_to_keep):
    modified_chunks = []

    for chunk in chunks:
        # Make a copy of the original metadata
        original_metadata = chunk.metadata
        print(f"original metadata: {original_metadata}")

        # Retain only the specified keys in the metadata
        modified_metadata = {key: original_metadata[key] for key in keys_to_keep if key in original_metadata}

        # Create a new dictionary with the 'source' key holding the modified_metadata
        new_metadata = {'source': modified_metadata}

        # Update the document with the modified metadata
        chunk.metadata = new_metadata
        print(f"modified metadata: {chunk.metadata}")

        # Add a comma between page_content and metadata
        original_chunk = chunk.page_content
        chunk.page_content = original_chunk + ', '
        print(f"chunk: {chunk}")

        modified_chunks.append(chunk)

    return modified_chunks


def print_vector_store_content(vector_store):
    # Access the stored embeddings and documents
    faiss_index = vector_store.index
    stored_docs = vector_store.docstore._dict  # Access the internal dictionary of documents

    # Create a mapping of document IDs to their integer indices
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(stored_docs.keys())}

    # Iterate through the documents and embeddings
    for doc_id, document in stored_docs.items():
        index = doc_id_to_index[doc_id]
        embedding = faiss_index.reconstruct(index)

        print(f"Document {doc_id}: {document}\n")
        print(f"Embedding {doc_id}: {embedding}\n")


def print_loaded_documents(documents):
    for document in documents:
        print(f'document: {document}')


def print_split_documents(documents):
    i = 0
    while i < len(documents):
        current_text = documents[i]
        print(f'current text: {current_text}')
        i += 1


def print_docs_for_question(documents):
    for document in documents:
        print(f'retrieved docs for prompt: {document}')
