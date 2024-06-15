'''
### MAIN ###
CHAT_MODEL_IN_SESSION_STATE = 'chat_model'

### FRONTEND ###
CHAT_HEADER = 'My Chatbot'
CHAT_GREETING = 'Hello 👋'
CHAT_INPUT = 'Type your question'
SPINNER_LABEL = 'Thinking...'

MESSAGES_KEY = 'messages'
CHAT_HISTORY_KEY = 'chathistory'
MESSAGE_CONTENT_KEY = 'content'

ROLE = 'role'
ROLE_ASSISTANT = 'assistant'
ROLE_USER = 'user'

USER_PROMPT_KEY = 'user_prompt'
TEXT_KEY = 'text'

DOCUMENT_PAGE_CONTENT_KEY = 'page_content'
DOCUMENT_SOURCE_KEY = 'source'
DOCUMENT_ID_KEY = 'id'
DOCUMENT_QUESTION_KEY = 'question'
DOCUMENT_QUESTION_ANSWER_KEY = 'answer'

CONTEXTUALIZED_SYSTEM_PROMPT = """
Your task is to transform a given question into a self-contained question. Follow these steps:

1. If the user's prompt refers to the previous chat history, rephrase it to be fully understandable on its own, without needing any prior context.
2. If the user's prompt is standalone and unrelated to the chat history, leave it unchanged.
3. Do NOT provide an answer to the question.
4. Do NOT provide any explanation of the question or why you did or did not rephrase it.
4. Focus solely on adjusting the wording if necessary.

Here is the chat history: {chathistory}

Here is the question: {user_prompt}

Self Contained Question:
"""

TEMPLATE_SYSTEM_PROMPT = """
User: {question}
Assistant: To answer the question, you should follow these steps:

1. Check if the provided context and chat history are non-empty.
2. Important: If context and chat history are both empty, respond with "I don't know the answer to that question based on the provided information."
3. If context or chat history (or both) are non-empty:
    a. Review the provided context carefully to see if it contains relevant information to answer the question.
    b. Also review the chat history to understand the context of the conversation and any relevant information provided in previous exchanges.
    c. If the context and chat history contain enough information to answer the question, provide a clear and concise answer based on that information.
    d. If the context and chat history do not contain enough information to answer the question, respond with "I don't know the answer to that question based on the provided information."
4. Do not make up answers or provide speculative information if the context and chat history do not contain relevant information to answer the question.\n

Here is the context:\n {context}

Here is the chat history:\n {chathistory}
"""

DOCUMENT_PROMPT_TEMPLATE = '{page_content}\n{source}\n'
DOCUMENT_PROMPT_VARIABLE_NAME = 'context'
QA_USER_PROMPT_KEY = 'question'

RESPONSE_ANSWER_KEY = 'answer'
RESPONSE_SOURCE_DOC_KEY = 'source_documents'
RESPONSE_SOURCES_KEY = 'sources'

FORMATTED_ANSWER_WITH_CONTEXT = '\n\n**Used context to answer your question:**\n\n'
FORMATTED_ANSWER_WITHOUT_CONTEXT = '\n\nThere are no documents that could be used to answer your question.'

### PREPROCESS DATASET ###
DATASET_EVAL = 'datasets/taylor_swift_test.csv'
DATASET = 'datasets/taylor_swift.csv'
REPLACEMENT_NAN_VALUES = 'unknown'

### INITIAL SETUP ###
HUGGING_FACE_API_KEY = 'HUGGING_FACE_API_KEY'
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'
SEARCH_TYPE_VALUE = 'similarity_score_threshold'
SCORE_THRESHOLD_KEY = 'score_threshold'
K_KEY = 'k'
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
LLM_TASK = 'text-generation'

'''