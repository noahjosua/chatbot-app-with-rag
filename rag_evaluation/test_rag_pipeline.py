import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, \
    AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

import constants.constants

dataset = EvaluationDataset()
dataset.pull(alias=constants.constants.DATASET_ALIAS)


@pytest.mark.parametrize(
    'test_case',
    dataset
)
def test_evaluate_rag_pipeline(test_case: LLMTestCase):
    # Retriever metrics
    contextual_recall_metric = ContextualRecallMetric()
    contextual_precision_metric = ContextualPrecisionMetric()
    contextual_relevancy_metric = ContextualRelevancyMetric()

    # Generator metrics
    answer_relevancy_metric = AnswerRelevancyMetric()
    faithfulness_metric = FaithfulnessMetric()

    assert_test(test_case, [contextual_recall_metric, contextual_precision_metric, contextual_relevancy_metric,
                            answer_relevancy_metric, faithfulness_metric])
