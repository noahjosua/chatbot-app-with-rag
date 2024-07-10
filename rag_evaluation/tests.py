from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, \
    AnswerRelevancyMetric, FaithfulnessMetric


def run_tests():
    # Initialize empty dataset object
    dataset = EvaluationDataset()

    # Pull from Confident
    dataset.pull(alias="Evaluation Dataset - Taylor Swift")

    # Retriever metrics
    contextual_recall_metric = ContextualRecallMetric()
    contextual_precision_metric = ContextualPrecisionMetric()
    contextual_relevancy_metric = ContextualRelevancyMetric()

    # Generator metrics
    answer_relevancy_metric = AnswerRelevancyMetric()
    faithfulness_metric = FaithfulnessMetric()

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[contextual_recall_metric, contextual_precision_metric, contextual_relevancy_metric,
                 answer_relevancy_metric, faithfulness_metric]
    )
