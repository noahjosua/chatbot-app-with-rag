import subprocess

from dotenv import load_dotenv

from constants import constants
from init import initialization
from rag_evaluation.dataset_for_confident_ai_generation import create_and_push_dataset_to_confident_ai
from rag_evaluation.json_dataset_generation import generate_json_for_eval

if __name__ == '__main__':
    chat_model_and_retriever = initialization.initial_setup(constants.DATASET_EVAL)
    chat_model = chat_model_and_retriever[0]
    retriever = chat_model_and_retriever[1]
    generate_json_for_eval(chat_model, retriever, constants.DATASET_EVAL, constants.EVALUATION_RESULTS)
    load_dotenv()
    create_and_push_dataset_to_confident_ai(constants.EVALUATION_RESULTS_PATH, constants.DATASET_ALIAS)
    subprocess.run(["deepeval", "test", "run", "test_rag_pipeline.py"], check=True)