# Chatbot-Application with RAG-Pipeline
As part of the elective course "Artificial Intelligence and Adaptive Systems",
a chatbot application with Retrieval Augmented Generation (RAG) pipeline has been developed.

The application was implemented using 
- Python (v3.12)
- LangChain (v0.2.7)
- Hugging Face
- Streamlit (v1.36.0)
- Deepeval (v0.21.65)

## System requirements
- Python v3.12
- IDE (e.g. PyCharm)

## Create virtual environment and install packages
1. Clone project and open it in IDE
2. Create virtual environment 
   - Settings --> Project --> Python Interpreter --> Add local Interpreter --> Virtual Environment 
   - Choose 'new', set Location to `...\chat-bot-with-rag\.venv` and Base Interpreter to where ever you have your Python Version installed, e.g. `...\AppData\Local\Programs\Python\Python312\python.exe`
3. Activate virtual environment in terminal: execute `.\.venv\Scripts\Activate` from root directory of project
4. Install packages in venv 
   - `pip install pipenv`
   - `pipenv install`

## Add .env file
Add .env file in root directory with following properties
  - HUGGING_FACE_API_KEY=***
  - OPENAI_API_KEY=*** (only necessary for executing evaluation)
  - CONFIDENT_API_KEY=*** (only necessary for executing evaluation)

## Grant access to LLM 
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## Run the application 
Run the `run_app.py` script


### How to get API Keys
#### Hugging Face
- [Login](https://huggingface.co/login)
- Profile --> Settings --> Access Tokens

### Open AI
- Login and go to [Playground](https://platform.openai.com/playground)
- Go to Tab 'API Keys'
- Make sure you have budget

### Confident AI
- [Sign up](https://app.confident-ai.com/auth/signup) and generate API Key