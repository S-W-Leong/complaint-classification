from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import text_preprocessing as _text_preprocessing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = ("Low", "Medium", "High")
CLASSICAL_MODEL_NAMES = (
    "multinomial_nb",
    "logistic_regression",
    "linear_svm_calibrated",
)
MIN_NARRATIVE_CHARACTERS = 50
MAX_NARRATIVE_CHARACTERS = 13_000
LIME_DISCLAIMER = "LIME is a local model approximation, not a causal explanation."


class NarrativeValidationError(ValueError):
    """Raised when text is outside the supported prototype range."""


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    display_name: str
    value_label: str


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    display_name: str
    predicted_label: str
    selected_value: float
    value_label: str
    class_values: Mapping[str, float]
    response_guidance: str


@dataclass(frozen=True)
class LimeExplanation:
    predicted_label: str
    terms: tuple[tuple[str, float], ...]
    disclaimer: str = LIME_DISCLAIMER


@dataclass(frozen=True)
class ClassicalResources:
    vectorizer: Any
    models: Mapping[str, Any]


@dataclass(frozen=True)
class TransformerResources:
    tokenizer: Any
    model: Any
    device: Any


MODEL_SPECS = {
    "multinomial_nb": ModelSpec(
        "multinomial_nb",
        "Multinomial Naive Bayes",
        "Model probability — not independently calibrated",
    ),
    "logistic_regression": ModelSpec(
        "logistic_regression",
        "Logistic Regression",
        "Model probability — not independently calibrated",
    ),
    "linear_svm_calibrated": ModelSpec(
        "linear_svm_calibrated",
        "Calibrated Linear SVM",
        "Calibrated probability",
    ),
    "distilbert": ModelSpec(
        "distilbert",
        "DistilBERT",
        "Softmax score — not calibrated",
    ),
}


def load_classical_resources(project_root: Path = PROJECT_ROOT) -> ClassicalResources:
    vectorizer = joblib.load(project_root / "artifacts/tfidf_vectorizer.joblib")
    models = {
        name: joblib.load(project_root / f"artifacts/models/{name}.joblib")
        for name in CLASSICAL_MODEL_NAMES
    }
    return ClassicalResources(vectorizer=vectorizer, models=models)


def classical_probabilities(
    texts: Sequence[str], model_name: str, resources: ClassicalResources
) -> np.ndarray:
    model = resources.models[model_name]
    values = model.predict_proba(resources.vectorizer.transform(list(texts)))
    indexes = [list(model.classes_).index(label) for label in LABEL_ORDER]
    return np.asarray(values)[:, indexes]


def mean_pool_document_logits(
    chunk_logits: Sequence[Sequence[float]],
    document_ids: Sequence[int],
    document_count: int,
) -> np.ndarray:
    logits = np.asarray(chunk_logits, dtype=float)
    return np.vstack(
        [
            logits[np.asarray(document_ids) == index].mean(axis=0)
            for index in range(document_count)
        ]
    )


def select_torch_device() -> Any:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_transformer_resources(
    project_root: Path = PROJECT_ROOT,
) -> TransformerResources:
    model_path = project_root / "artifacts/models/distilbert"
    device = select_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return TransformerResources(tokenizer=tokenizer, model=model, device=device)


def transformer_probabilities(
    texts: Sequence[str],
    resources: TransformerResources,
    batch_size: int = 16,
) -> np.ndarray:
    encoded = resources.tokenizer(
        list(texts),
        truncation=True,
        max_length=256,
        stride=64,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    document_ids = encoded.pop("overflow_to_sample_mapping").tolist()
    chunk_logits = []
    resources.model.eval()
    with torch.no_grad():
        for start in range(0, len(document_ids), batch_size):
            inputs = {
                key: values[start : start + batch_size].to(resources.device)
                for key, values in encoded.items()
            }
            chunk_logits.extend(resources.model(**inputs).logits.cpu().tolist())
    pooled = mean_pool_document_logits(chunk_logits, document_ids, len(texts))
    pooled -= pooled.max(axis=1, keepdims=True)
    values = np.exp(pooled)
    return values / values.sum(axis=1, keepdims=True)


def predict_one(
    narrative: str,
    model_name: str,
    classical: ClassicalResources,
    transformer: TransformerResources | None = None,
) -> PredictionResult:
    narrative = validate_narrative(narrative)
    if model_name == "distilbert":
        if transformer is None:
            raise ValueError("DistilBERT resources are required for this model.")
        values = transformer_probabilities([narrative], transformer)[0]
    else:
        values = classical_probabilities([narrative], model_name, classical)[0]
    return build_prediction_result(model_name, values)


def compare_all(
    narrative: str,
    classical: ClassicalResources,
    transformer: TransformerResources,
) -> list[PredictionResult]:
    narrative = validate_narrative(narrative)
    results = []
    for model_name in MODEL_SPECS:
        if model_name == "distilbert":
            values = transformer_probabilities([narrative], transformer)[0]
        else:
            values = classical_probabilities([narrative], model_name, classical)[0]
        results.append(build_prediction_result(model_name, values))
    return results


def generate_lime_explanation(
    narrative: str,
    prediction: PredictionResult,
    probability_function: Callable[[Sequence[str]], np.ndarray],
) -> LimeExplanation:
    narrative = validate_narrative(narrative)
    label_index = LABEL_ORDER.index(prediction.predicted_label)
    explainer = LimeTextExplainer(
        class_names=list(LABEL_ORDER), random_state=20_260_815
    )
    explanation = explainer.explain_instance(
        narrative,
        probability_function,
        labels=[label_index],
        num_features=10,
        num_samples=250,
    )
    return LimeExplanation(
        predicted_label=prediction.predicted_label,
        terms=tuple(
            (term, float(weight))
            for term, weight in explanation.as_list(label=label_index)
        ),
    )


def load_test_macro_f1(project_root: Path = PROJECT_ROOT) -> dict[str, float]:
    report_path = project_root / "data/reports/phase6_test_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        name: float(report["models"][name]["metrics"]["macro_f1"])
        for name in MODEL_SPECS
    }


def validate_narrative(value: str) -> str:
    narrative = value.strip()
    if not narrative:
        raise NarrativeValidationError(
            "Enter a complaint narrative before running a prediction."
        )
    if len(narrative) < MIN_NARRATIVE_CHARACTERS:
        raise NarrativeValidationError(
            "Enter at least 50 characters so the narrative resembles the development data."
        )
    if len(narrative) > MAX_NARRATIVE_CHARACTERS:
        raise NarrativeValidationError(
            "Use 13,000 characters or fewer for this academic prototype."
        )
    return narrative


def model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_name}") from error


def response_guidance(label: str) -> str:
    return {
        "Low": "Illustrative target: standard review within 5 business days.",
        "Medium": "Illustrative target: priority review within 2 business days.",
        "High": "Illustrative target: immediate escalation and same-business-day human review.",
    }[label]


def build_prediction_result(
    model_name: str, probabilities: Sequence[float]
) -> PredictionResult:
    if len(probabilities) != len(LABEL_ORDER):
        raise ValueError("Expected one value for each urgency class.")
    class_values = {
        label: float(value) for label, value in zip(LABEL_ORDER, probabilities)
    }
    predicted_label = max(class_values, key=class_values.get)
    spec = model_spec(model_name)
    return PredictionResult(
        model_name=model_name,
        display_name=spec.display_name,
        predicted_label=predicted_label,
        selected_value=class_values[predicted_label],
        value_label=spec.value_label,
        class_values=class_values,
        response_guidance=response_guidance(predicted_label),
    )
