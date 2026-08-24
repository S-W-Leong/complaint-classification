"""Immutable product-state operations for the complaint workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from src.prototype_inference import PredictionResult


Priority = Literal["Low", "Medium", "High"]
CaseStatus = Literal["New", "In review", "Escalated", "Resolved"]
CaseView = Literal["queue", "my_cases", "escalations"]
CURRENT_OFFICER = "Alex Peterson"


def _require_timezone_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class TimelineEvent:
    title: str
    detail: str
    occurred_at: datetime


@dataclass(frozen=True)
class ComplaintCase:
    case_id: str
    narrative: str
    issue_summary: str
    category: str
    product: str
    source: str
    received_at: datetime
    response_due: datetime
    ai_priority: Priority
    ai_score: float
    score_label: str
    recommended_action: str
    explanation_terms: tuple[tuple[str, float], ...]
    assignee: str | None
    status: CaseStatus
    timeline: tuple[TimelineEvent, ...]


def recommended_action(priority: Priority) -> str:
    return {
        "Low": "Complete a standard review within 5 business days.",
        "Medium": "Prioritize review and respond within 2 business days.",
        "High": "Escalate now and complete a same-day priority review.",
    }[priority]


def _response_deadline(received_at: datetime, priority: Priority) -> datetime:
    if priority == "High":
        return received_at

    business_days = {"Medium": 2, "Low": 5}[priority]
    deadline = received_at
    while business_days:
        deadline += timedelta(days=1)
        if deadline.weekday() < 5:
            business_days -= 1
    return deadline


def ordered_case_ids(cases: Sequence[ComplaintCase]) -> list[str]:
    return [case.case_id for case in sorted(cases, key=lambda item: item.received_at, reverse=True)]


def _case(
    case_id: str,
    narrative: str,
    summary: str,
    category: str,
    product: str,
    source: str,
    received_at: datetime,
    priority: Priority,
    score: float,
    assignee: str | None = None,
    status: CaseStatus = "New",
    terms: tuple[tuple[str, float], ...] = (),
) -> ComplaintCase:
    event = TimelineEvent("Complaint received", f"Received via {source}", received_at)
    return ComplaintCase(
        case_id=case_id,
        narrative=narrative,
        issue_summary=summary,
        category=category,
        product=product,
        source=source,
        received_at=received_at,
        response_due=_response_deadline(received_at, priority),
        ai_priority=priority,
        ai_score=score,
        score_label="Softmax score — not calibrated",
        recommended_action=recommended_action(priority),
        explanation_terms=terms,
        assignee=assignee,
        status=status,
        timeline=(event,),
    )


def seed_cases(now: datetime) -> list[ComplaintCase]:
    """Return deterministic synthetic records, with ``now`` as the reference clock."""
    _require_timezone_aware(now, "now")
    records = (
        ("CMP-2026-0184", "An unauthorized card transaction remains on my account and the funds have not been returned.", "Unauthorized card transaction", "Unauthorized transactions", "Debit card", "Web form", 0, "High", .91, CURRENT_OFFICER, "New", (("unauthorized", .82), ("card", .66))),
        ("CMP-2026-0183", "My mortgage servicer has not corrected the payment history after multiple requests.", "Mortgage payment history is incorrect", "Credit reporting", "Mortgage", "Phone", 1, "Medium", .64, CURRENT_OFFICER, "In review", (("payment", .54), ("history", .48))),
        ("CMP-2026-0182", "A debt collector sent a letter about an account I do not recognize.", "Debt collection account is not recognized", "Debt collection", "Debt collection", "Mail", 2, "High", .86, None, "Escalated", (("collector", .61), ("recognize", .55))),
        ("CMP-2026-0181", "I would like a copy of my credit report and information about how to dispute an error.", "Request for credit report dispute information", "Credit reporting", "Credit report", "Web form", 3, "Low", .23, CURRENT_OFFICER, "New", ()),
        ("CMP-2026-0180", "The bank charged an overdraft fee even though I deposited money before the transaction.", "Overdraft fee appears incorrect", "Bank account or service", "Checking account", "Mobile app", 4, "Medium", .57, None, "New", (("overdraft", .49),)),
        ("CMP-2026-0179", "My account was closed without an explanation and I need help understanding what happened.", "Account closed without explanation", "Bank account or service", "Checking account", "Email", 5, "High", .79, CURRENT_OFFICER, "Escalated", (("closed", .63),)),
        ("CMP-2026-0178", "The annual fee on my credit card statement is higher than the fee described when I opened it.", "Credit card annual fee is unexpected", "Credit card", "Credit card", "Web form", 6, "Medium", .52, None, "Resolved", ()),
        ("CMP-2026-0177", "I need help updating my mailing address and understanding a routine statement question.", "Routine statement and address request", "Bank account or service", "Checking account", "Branch", 7, "Low", .14, None, "New", ()),
    )
    return [
        _case(case_id, narrative, summary, category, product, source, now - timedelta(hours=hours), priority, score, assignee, status, terms)
        for case_id, narrative, summary, category, product, source, hours, priority, score, assignee, status, terms in records
    ]


def filter_cases(
    cases: Sequence[ComplaintCase],
    *,
    view: CaseView = "queue",
    query: str = "",
    priority: Priority | None = None,
    status: CaseStatus | None = None,
) -> list[ComplaintCase]:
    needle = query.strip().lower()
    filtered: list[ComplaintCase] = []
    for case in cases:
        if view == "queue" and case.status == "Resolved":
            continue
        if view == "my_cases" and case.assignee != CURRENT_OFFICER:
            continue
        if view == "escalations" and case.status != "Escalated":
            continue
        if priority is not None and case.ai_priority != priority:
            continue
        if status is not None and case.status != status:
            continue
        haystack = " ".join((case.case_id, case.issue_summary, case.category, case.product, case.narrative)).lower()
        if needle and needle not in haystack:
            continue
        filtered.append(case)
    return filtered


def next_case_id(cases: Sequence[ComplaintCase]) -> str:
    highest = max((int(case.case_id.rsplit("-", 1)[1]) for case in cases), default=0)
    return f"CMP-2026-{highest + 1:04d}"


def create_case(
    narrative: str,
    category: str,
    product: str,
    source: str,
    prediction: PredictionResult,
    existing_cases: Sequence[ComplaintCase],
    at: datetime,
) -> ComplaintCase:
    from src.prototype_inference import validate_narrative

    _require_timezone_aware(at, "at")
    narrative = validate_narrative(narrative)
    priority = prediction.predicted_label
    if priority not in ("Low", "Medium", "High"):
        raise ValueError(f"Unsupported prediction priority: {priority}")
    summary = narrative.strip().split(".", 1)[0]
    if len(summary) > 100:
        summary = summary[:97].rstrip() + "..."
    event = TimelineEvent("Complaint received", f"Received via {source}", at)
    return ComplaintCase(
        case_id=next_case_id(existing_cases), narrative=narrative, issue_summary=summary,
        category=category, product=product, source=source, received_at=at,
        response_due=_response_deadline(at, priority), ai_priority=priority,
        ai_score=prediction.selected_value, score_label=prediction.value_label,
        recommended_action=recommended_action(priority), explanation_terms=(),
        assignee=None, status="New", timeline=(event,),
    )


def with_explanation(
    case: ComplaintCase,
    terms: Sequence[tuple[str, float]],
) -> ComplaintCase:
    cleaned = tuple((term, float(weight)) for term, weight in terms if weight != 0)
    return replace(case, explanation_terms=cleaned)


def assign_case(case: ComplaintCase, assignee: str, at: datetime) -> ComplaintCase:
    _require_timezone_aware(at, "at")
    event = TimelineEvent("Case assigned", f"Assigned to {assignee}", at)
    return replace(case, assignee=assignee, timeline=case.timeline + (event,))


def add_note(case: ComplaintCase, note: str, author: str, at: datetime) -> ComplaintCase:
    _require_timezone_aware(at, "at")
    cleaned = note.strip()
    if not cleaned:
        raise ValueError("Enter an internal note before saving.")
    event = TimelineEvent("Internal note added", f"{author}: {cleaned}", at)
    return replace(case, timeline=case.timeline + (event,))


def set_case_status(case: ComplaintCase, status: CaseStatus, actor: str, at: datetime) -> ComplaintCase:
    _require_timezone_aware(at, "at")
    title = "Case escalated" if status == "Escalated" else f"Status changed to {status}"
    event = TimelineEvent(title, f"Updated by {actor}", at)
    return replace(case, status=status, timeline=case.timeline + (event,))
