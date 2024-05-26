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

DOCUMENT_PAGE_CONTENT_KEY = 'page_content'
DOCUMENT_SOURCE_KEY = 'source'
DOCUMENT_SHOW_ID_KEY = 'show_id'

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

TEMPLATE_SYSTEM_PROMPT = """
User: {question}
Assistant: Use the following pieces of retrieved context as well as the chat history to answer the question 
(which might reference context in the chat history). If you don't know the answer, 
just say that you don't know, don't try to make up an answer. 
If you don't get context, don't refer to it in your answer. Keep the answer as concise as possible.

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
DATASET = 'netflix_titles.csv'
REPLACEMENT_NAN_VALUES = 'unknown'

### INITIAL SETUP ###
HUGGING_FACE_API_KEY = 'HUGGING_FACE_API_KEY'
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'
SEARCH_TYPE = 'similarity'
K_KEY = 'k'
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
LLM_TASK = 'text-generation'
