### MAIN ###
CHAT_MODEL_IN_SESSION_STATE = 'chat_model'

### FRONTEND ###
CHAT_HEADER = 'My Chatbot'
CHAT_GREETING = 'Hello 👋'
CHAT_INPUT = 'Type your question'
SPINNER_LABEL = 'Thinking...'

MESSAGES_KEY = 'messages'
CHAT_HISTORY_KEY = 'chat_history'
MESSAGE_CONTENT_KEY = 'content'

ROLE = 'role'
ROLE_ASSISTANT = 'assistant'
ROLE_USER = 'user'

USER_PROMPT_KEY = 'user_prompt'
TEXT_KEY = 'text'

DOCUMENT_DOCUMENT_TITLE_KEY = 'document_title'
DOCUMENT_PAGE_CONTENT_KEY = 'page_content'
DOCUMENT_SOURCE_KEY = 'source'
DOCUMENT_ID_KEY = 'id'
DOCUMENT_QUESTION_KEY = 'question'
DOCUMENT_QUESTION_ANSWER_KEY = 'answer'

CONTEXTUALIZED_SYSTEM_PROMPT = """
You will receive a question and a chat history.
Your task is to rephrase the question in a way that makes it comprehensible without needing the chat history context, 
if the original question references the chat history.
If the question does not reference the chat history, you should return the original question unchanged.
Do not attempt to answer the question. Only rephrase it if necessary based on the criteria above.

Here is the chat history: {chat_history}
Here is the original question: {user_prompt}

Please provide the rephrased question or the original question if no rephrasing is needed.
"""

'''
TEMPLATE_SYSTEM_PROMPT = """
User: {question}
Assistant: Use the following pieces of retrieved context as well as the chat history to answer the question 
(which might reference context in the chat history). If you don't know the answer, 
just say that you don't know, don't try to make up an answer. 
If you don't get context, don't refer to it in your answer. Keep the answer as concise as possible.

Here is the context: {context}
Here is the chat history: {chat_history}
"""
'''
TEMPLATE_SYSTEM_PROMPT = """
User: {question}
Assistant: Use the following pieces of retrieved context as well as the chat history to answer the question 
(which might reference context in the chat history).
To answer the question, you should:
1. Review the provided context carefully to see if it contains relevant information to answer the question.
2. Also review the chat history to understand the context of the conversation and any relevant information provided in previous exchanges.
3. If the context and chat history contain enough information to answer the question, provide a clear and concise answer based on that information.
4. If the context and chat history do not contain enough information to answer the question, respond with "I don't know the answer to that question based on the provided information."
5. Do not make up answers or provide speculative information if the context and chat history do not contain relevant information to answer the question.
Here is the context: {context}
Here is the chat history: {chat_history}
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
DATASET_EVAL = 'lamini_taylor_swift_test.csv'
DATASET = 'lamini_taylor_swift_train.csv'
REPLACEMENT_NAN_VALUES = 'unknown'

### INITIAL SETUP ###
HUGGING_FACE_API_KEY = 'HUGGING_FACE_API_KEY'
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'
SEARCH_TYPE = 'similarity'
K_KEY = 'k'
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
LLM_TASK = 'text-generation'

### Evaluation ###
LLM_MODEL_NAME_EVAL = 'google/flan-t5-base'
LLM_TASK_EVAL = 'text2text-generation'
