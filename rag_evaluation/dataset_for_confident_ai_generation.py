import os

import deepeval
import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase


def create_and_push_dataset_to_confident_ai():
    api_key = os.getenv("CONFIDENT_API_KEY")
    deepeval.login_with_confident_api_key(api_key)

    df = pd.read_json('evaluation_results.json')

    test_cases = []
    for index, row in df.iterrows():
        question = row['question']
        expected_answer = row['expected_answer']
        actual_answer = row['actual_answer']
        context = row['context']

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=[context]
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    dataset.push(alias="Evaluation Dataset - Taylor Swift", overwrite=False)
