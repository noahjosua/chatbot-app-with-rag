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
