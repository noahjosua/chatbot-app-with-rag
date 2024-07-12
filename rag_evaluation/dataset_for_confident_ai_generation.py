import os

import deepeval
import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

from constants import constants


def create_and_push_dataset_to_confident_ai(json_file_path, dataset_alias):
    api_key = os.getenv(constants.CONFIDENT_API_KEY)
    deepeval.login_with_confident_api_key(api_key)

    df = pd.read_json(json_file_path)

    test_cases = []
    for index, row in df.iterrows():
        question = row[constants.QUESTION_KEY]
        expected_answer = row[constants.EXPECTED_ANSWER_KEY]
        actual_answer = row[constants.ACTUAL_ANSWER_KEY]
        context = row[constants.CONTEXT_KEY]

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=[context]
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    dataset.push(alias=dataset_alias, overwrite=False)
