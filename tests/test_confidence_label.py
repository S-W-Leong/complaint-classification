from datetime import datetime, timezone

from src.complaint_operations import seed_cases
from src.prototype_inference import MODEL_SPECS


def test_distilbert_score_is_labeled_as_uncalibrated_confidence() -> None:
    assert MODEL_SPECS["distilbert"].value_label == "Softmax confidence — not calibrated"


def test_seeded_case_uses_confidence_label() -> None:
    case = seed_cases(datetime(2026, 9, 12, tzinfo=timezone.utc))[0]

    assert case.score_label == "Softmax confidence — not calibrated"
