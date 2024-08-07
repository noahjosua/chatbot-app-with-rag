### DATASET ###
DATASET_ORIGINAL = 'datasets/original/taylor_swift.csv'
DATASET_TEST = 'datasets/test/taylor_swift_test.csv'
DATASET_EVAL = '../datasets/eval/taylor_swift_eval.csv'

### HUGGING FACE ###
HUGGING_FACE_API_KEY = 'HUGGING_FACE_API_KEY'

### CONFIDENT AI ###
CONFIDENT_API_KEY = 'CONFIDENT_API_KEY'

### LLM ###
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
LLM_TASK = 'text-generation'

### EMBEDDINGS MODEL ###
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'

### RETRIEVER ###
SEARCH_TYPE_VALUE = 'similarity_score_threshold'
SCORE_THRESHOLD_KEY = 'score_threshold'
K_KEY = 'k'

### MAIN ###
CHAT_MODEL_IN_SESSION_STATE = 'chat_model'

### FRONTEND ###
CHAT_HEADER = 'Ask me anything about Taylor Swift!'
CHAT_GREETING = 'Hello 👋'
CHAT_INPUT = 'Type your question'
SPINNER_LABEL = 'Thinking...'

### SESSION STATE ###
MESSAGES_KEY = 'messages'
CHAT_HISTORY_KEY = 'chathistory'
MESSAGE_CONTENT_KEY = 'content'

ROLE = 'role'
ROLE_ASSISTANT = 'assistant'
ROLE_USER = 'user'

### DOCUMENT ###
DOCUMENT_ID_KEY = 'id'
DOCUMENT_SOURCE_KEY = 'source'
DOCUMENT_PAGE_CONTENT_KEY = 'page_content'
UNUSABLE_ROW_KEY = 'having trouble understanding'

### REPHRASE USER PROMPT ###
REPHRASED_SYSTEM_PROMPT = """
Please rephrase the following question to make it more concise and understandable. 
Ensure that the key information and intent of the original question are preserved.

Here is the question:\n {user_prompt}

Rephrased Question: \n
"""

USER_PROMPT_KEY = 'user_prompt'
TEXT_KEY = 'text'

### CHAT FLOW ###
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

QUESTION_KEY = 'question'
CONTEXT_KEY = 'context'
DOCUMENT_PROMPT_TEMPLATE = '{page_content}\n{source}\n'

ANSWER_KEY = 'answer'
RESPONSE_SOURCE_DOC_KEY = 'source_documents'
RESPONSE_SOURCES_KEY = 'sources'

FORMATTED_ANSWER_WITH_CONTEXT = '\n\n**Used context to answer your question:**\n\n'
FORMATTED_ANSWER_WITHOUT_CONTEXT = '\n\nThere are no documents that could be used to answer your question.'

### EVALUATION ###
TEMPLATE_SYSTEM_PROMPT_FOR_EVAL = """
User: {question}
Assistant: To answer the question, you should follow these steps:

1. Check if the provided context is not empty.
2. Important: If context is empty, respond with "I don't know the answer to that question based on the provided information."
3. If context is not empty:
    a. Review the provided context carefully to see if it contains relevant information to answer the question.
    b. If the context contains enough information to answer the question, provide a clear and concise answer based on that information.
    c. If the context does not contain enough information to answer the question, respond with "I don't know the answer to that question based on the provided information."
4. Do not make up answers or provide speculative information if the context does not contain relevant information to answer the question.\n

Here is the context:\n {context}
"""

EVALUATION_RESULTS = 'evaluation_results.json'
EVALUATION_RESULTS_PATH = '../rag_evaluation/evaluation_results.json'
DATASET_ALIAS = 'Evaluation Dataset - Taylor Swift'

QUESTION_ID_KEY = 'question_id'
ACTUAL_ANSWER_KEY = 'actual_answer'
EXPECTED_ANSWER_KEY = 'expected_answer'
