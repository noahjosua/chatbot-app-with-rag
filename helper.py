import os


def modify_metadata(chunks):
    for chunk in chunks:
        lines = chunk.page_content.split('\\n')
        show_id = lines[0].split(': ')[1].split('\n')[0].strip()
        chunk.metadata['source'] = f"document: '{os.path.basename(chunk.metadata['source'])}'"
        chunk.metadata['show_id'] = show_id
        del chunk.metadata['row']
    return chunks


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
