from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import streamlit as st

from src.complaint_operations import (
    CURRENT_OFFICER,
    ComplaintCase,
    add_note,
    assign_case,
    create_case,
    filter_cases,
    ordered_case_ids,
    seed_cases,
    set_case_status,
    with_explanation,
)
from src.prototype_inference import (
    ClassicalResources,
    NarrativeValidationError,
    generate_lime_explanation,
    load_transformer_resources,
    predict_one,
    transformer_probabilities,
    validate_narrative,
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
VIEW_LABELS = {
    "queue": "Complaint queue",
    "my_cases": "My cases",
    "escalations": "Escalations",
}
PRIORITY_OPTIONS = ("All priorities", "High", "Medium", "Low")
STATUS_OPTIONS = ("All statuses", "New", "In review", "Escalated", "Resolved")
# Keep the action column wide enough for the arrow icon and the "Open" label
# at the browser's default zoom level.
QUEUE_COLUMN_WIDTHS = (3.15, 1.1, 1.2, 1.2, 1.8, 1.05)
NEW_COMPLAINT_CATEGORIES = (
    "Billing / Transactions",
    "Credit reporting",
    "Unauthorized transactions",
    "Debt collection",
    "Bank account or service",
)
NEW_COMPLAINT_PRODUCTS = (
    "Credit card",
    "Checking account",
    "Credit report",
    "Mortgage",
    "Debt collection",
)
NEW_COMPLAINT_SOURCES = ("Web form", "Phone", "Email", "Mail", "Mobile app", "Branch")
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def cached_transformer_resources():
    return load_transformer_resources()


def initialize_state() -> None:
    if "cases" not in st.session_state:
        st.session_state["cases"] = seed_cases(datetime.now(timezone.utc))
    st.session_state.setdefault("screen", "queue")
    st.session_state.setdefault("active_view", "queue")
    st.session_state.setdefault("selected_case_id", None)


def navigate(
    screen: str, *, case_id: str | None = None, view: str | None = None
) -> None:
    st.session_state["screen"] = screen
    if case_id is not None:
        st.session_state["selected_case_id"] = case_id
    if view is not None:
        st.session_state["active_view"] = view


def replace_case(updated: ComplaintCase) -> None:
    st.session_state["cases"] = [
        updated if case.case_id == updated.case_id else case
        for case in st.session_state["cases"]
    ]


def create_and_triage_case(
    narrative: str,
    category: str,
    product: str,
    source: str,
) -> ComplaintCase | None:
    try:
        cleaned = validate_narrative(narrative)
        transformer = cached_transformer_resources()
        prediction = predict_one(
            cleaned, "distilbert", ClassicalResources(None, {}), transformer
        )
        created = create_case(
            narrative=cleaned,
            category=category,
            product=product,
            source=source,
            prediction=prediction,
            existing_cases=st.session_state["cases"],
            at=datetime.now(timezone.utc),
        )
    except NarrativeValidationError as error:
        st.error(str(error))
        return None
    except Exception:
        LOGGER.exception("Case triage failed")
        st.error("AI triage is temporarily unavailable. Your complaint has not been created.")
        return None
    st.session_state["cases"] = [created, *st.session_state["cases"]]
    navigate("case", case_id=created.case_id, view="queue")
    return created


def generate_case_explanation(case: ComplaintCase) -> ComplaintCase | None:
    try:
        transformer = cached_transformer_resources()
        prediction = predict_one(
            case.narrative, "distilbert", ClassicalResources(None, {}), transformer
        )
        explanation = generate_lime_explanation(
            case.narrative,
            prediction,
            lambda texts: transformer_probabilities(texts, transformer),
        )
        updated = with_explanation(case, explanation.terms)
    except Exception:
        LOGGER.exception("Case explanation failed")
        st.error("An explanation is temporarily unavailable. Please try again.")
        return None
    replace_case(updated)
    return updated


def load_product_css() -> None:
    css = (ASSETS_DIR / "complaint_operations.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _set_view(view: str) -> None:
    navigate("queue", view=view)


def _reset_queue_filters() -> None:
    st.session_state["queue_search"] = ""
    st.session_state["queue_priority"] = "All priorities"
    st.session_state["queue_status"] = "All statuses"


def render_sidebar() -> None:
    with st.container(key="hover_sidebar"):
        st.markdown("### Complaint operations")
        st.caption("Daily triage workspace")
        st.button(
            "Queue",
            key="nav_queue",
            icon=":material/inbox:",
            width="stretch",
            on_click=_set_view,
            args=("queue",),
        )
        st.button(
            "My cases",
            key="nav_my_cases",
            icon=":material/person:",
            width="stretch",
            on_click=_set_view,
            args=("my_cases",),
        )
        st.button(
            "Escalations",
            key="nav_escalations",
            icon=":material/priority_high:",
            width="stretch",
            on_click=_set_view,
            args=("escalations",),
        )
        st.button(
            "Reports",
            key="nav_reports",
            icon=":material/assessment:",
            width="stretch",
            on_click=navigate,
            args=("reports",),
        )
        st.divider()
        st.caption("Signed in as")
        st.markdown(f"**{CURRENT_OFFICER}**")
        st.caption("Triage officer")


def _selected_filter(value: str, all_value: str) -> str | None:
    return None if value == all_value else value


def _active_filtered_cases() -> list[ComplaintCase]:
    return filter_cases(
        st.session_state["cases"],
        view=st.session_state["active_view"],
        query=st.session_state.get("queue_search", ""),
        priority=_selected_filter(
            st.session_state.get("queue_priority", "All priorities"),
            "All priorities",
        ),
        status=_selected_filter(
            st.session_state.get("queue_status", "All statuses"),
            "All statuses",
        ),
    )


def _format_received(case: ComplaintCase) -> str:
    return case.received_at.strftime("%d %b at %H:%M UTC")


def _format_due(case: ComplaintCase) -> str:
    return case.response_due.strftime("%d %b at %H:%M UTC")


def selected_case() -> ComplaintCase:
    case_id = st.session_state["selected_case_id"]
    return next(case for case in st.session_state["cases"] if case.case_id == case_id)


def _navigate_case(case_id: str) -> None:
    navigate("case", case_id=case_id)


def _case_mutation_time() -> datetime:
    return datetime.now(timezone.utc)


def _assign_selected_case() -> None:
    assignee = st.session_state["case_assignee"]
    if assignee is None:
        return
    updated = assign_case(selected_case(), assignee, _case_mutation_time())
    replace_case(updated)
    st.toast(f"Assigned to {assignee}.", icon=":material/check:")


def _set_selected_case_status(status: str | None = None) -> None:
    target_status = status or st.session_state["case_status"]
    updated = set_case_status(
        selected_case(), target_status, CURRENT_OFFICER, _case_mutation_time()
    )
    replace_case(updated)
    st.session_state["case_status"] = target_status
    st.toast(f"Status updated to {target_status}.", icon=":material/check:")


def _add_internal_note() -> None:
    try:
        updated = add_note(
            selected_case(),
            st.session_state["internal_note"],
            CURRENT_OFFICER,
            _case_mutation_time(),
        )
    except ValueError as error:
        st.toast(str(error), icon=":material/error:")
        return
    replace_case(updated)
    st.session_state["internal_note"] = ""
    st.toast("Internal note added.", icon=":material/check:")


def _sync_case_controls(case: ComplaintCase) -> None:
    if st.session_state.get("case_controls_for") != case.case_id:
        st.session_state["case_assignee"] = None
        st.session_state["case_status"] = case.status
        st.session_state["case_controls_for"] = case.case_id


def render_case_header(case: ComplaintCase, ordered_ids: list[str], index: int) -> None:
    previous_id = ordered_ids[index - 1] if index else None
    next_id = ordered_ids[index + 1] if index + 1 < len(ordered_ids) else None
    breadcrumb_column, previous_column, position_column, next_column = st.columns(
        (7, 1.4, 0.8, 1.2)
    )
    with breadcrumb_column:
        st.caption(f"{VIEW_LABELS[st.session_state['active_view']]} / Case workspace")
    with previous_column:
        st.button(
            "Previous case",
            key="previous_case",
            icon=":material/arrow_back:",
            help="Previous case",
            disabled=previous_id is None,
            width="stretch",
            on_click=_navigate_case,
            args=(previous_id,) if previous_id else (),
        )
    with position_column:
        st.caption(f"{index + 1} of {len(ordered_ids)}" if ordered_ids else "Not in queue")
    with next_column:
        st.button(
            "Next case",
            key="next_case",
            icon=":material/arrow_forward:",
            help="Next case",
            disabled=next_id is None,
            width="stretch",
            on_click=_navigate_case,
            args=(next_id,) if next_id else (),
        )
    st.divider()


def render_case_identity(case: ComplaintCase) -> None:
    st.markdown(f"# {case.case_id}")
    st.caption(
        f"{case.category} | {case.product} | Received {_format_received(case)}"
    )
    st.divider()


def render_case_summary(case: ComplaintCase) -> None:
    st.markdown("### Issue summary")
    st.markdown(case.issue_summary)
    st.caption(f"Source: {case.source}")
    st.markdown("### Complaint narrative")
    st.markdown(case.narrative)


def render_timeline(case: ComplaintCase) -> None:
    st.markdown("### Activity timeline")
    for event in reversed(case.timeline):
        st.markdown(f"**{event.title}**  \n{event.detail}")
        st.caption(event.occurred_at.strftime("%d %b at %H:%M UTC"))


def render_note_form() -> None:
    st.markdown("### Add internal note")
    st.text_area(
        "Internal note",
        key="internal_note",
        placeholder="Record a contact, decision, or next step.",
    )
    st.button(
        "Add note",
        key="add_note",
        icon=":material/note_add:",
        on_click=_add_internal_note,
    )


def render_case_actions(case: ComplaintCase) -> None:
    _sync_case_controls(case)
    with st.container(border=True, key="ai_guidance"):
        st.markdown("### AI priority")
        st.markdown(
            f"<span class='priority priority-{case.ai_priority.lower()}'>{case.ai_priority}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Priority score:** {case.ai_score:.0%}")
        st.divider()
        st.markdown("### Why this was flagged")
        if st.button(
            "Generate explanation",
            key="generate_explanation",
            icon=":material/lightbulb:",
        ):
            updated = generate_case_explanation(case)
            if updated is not None:
                case = updated
        if case.explanation_terms:
            st.caption("Weighted influencing terms")
            for term, weight in case.explanation_terms:
                st.markdown(f"**{term}** ({weight:+.2f})")
        else:
            st.caption("No additional review signals are available for this complaint.")
        st.divider()
        st.markdown("### Recommended action")
        st.markdown(case.recommended_action)
    st.markdown("### Case details")
    st.markdown(f"**Current assignee:** {case.assignee or 'Unassigned'}")
    st.selectbox(
        "Assign to",
        (CURRENT_OFFICER,),
        key="case_assignee",
        index=None,
        placeholder="Assign a case owner",
        on_change=_assign_selected_case,
    )
    st.markdown(f"**Response due:** {_format_due(case)}")
    st.selectbox(
        "Status",
        STATUS_OPTIONS[1:],
        key="case_status",
        on_change=_set_selected_case_status,
    )
    st.markdown("### Case actions")
    st.button(
        "Mark in review",
        key="mark_in_review",
        icon=":material/rate_review:",
        on_click=_set_selected_case_status,
        args=("In review",),
    )
    st.button(
        "Escalate case",
        key="escalate_case",
        icon=":material/priority_high:",
        type="primary",
        on_click=_set_selected_case_status,
        args=("Escalated",),
    )


def render_new_complaint() -> None:
    st.header("New complaint")
    st.caption("Capture the complaint details and set an operational priority.")
    with st.form("new_complaint"):
        narrative = st.text_area(
            "Complaint narrative",
            key="new_narrative",
            placeholder="Describe what happened, when it happened, and the outcome needed.",
            height=220,
        )
        category = st.selectbox(
            "Category", NEW_COMPLAINT_CATEGORIES, key="new_category"
        )
        product = st.selectbox("Product", NEW_COMPLAINT_PRODUCTS, key="new_product")
        source = st.selectbox("Source", NEW_COMPLAINT_SOURCES, key="new_source")
        submitted = st.form_submit_button(
            "Create and triage case",
            key="create_case",
            type="primary",
            icon=":material/add:",
        )
    if submitted:
        with st.spinner("Creating and prioritizing case…"):
            create_and_triage_case(narrative, category, product, source)


def render_case_workspace() -> None:
    case = selected_case()
    ordered_ids = ordered_case_ids(_active_filtered_cases())
    if case.case_id not in ordered_ids:
        ordered_ids = []
        index = 0
    else:
        index = ordered_ids.index(case.case_id)
    render_case_header(case, ordered_ids, index)
    narrative_column, action_column = st.columns([2.1, 1], gap="large")
    with narrative_column:
        render_case_identity(case)
        render_case_summary(case)
        render_timeline(case)
        render_note_form()
    with action_column:
        render_case_actions(case)


def render_queue() -> None:
    active_view = st.session_state["active_view"]
    st.header(VIEW_LABELS[active_view])
    st.caption("Review incoming complaints and act on the next priority.")
    st.button(
        "New complaint",
        key="new_complaint",
        icon=":material/add:",
        type="primary",
        on_click=navigate,
        args=("new",),
    )

    search_column, priority_column, status_column = st.columns((2, 1, 1))
    with search_column:
        st.text_input(
            "Search cases",
            placeholder="Search by case ID, issue, or product",
            key="queue_search",
        )
    with priority_column:
        st.selectbox(
            "AI priority", PRIORITY_OPTIONS, key="queue_priority"
        )
    with status_column:
        st.selectbox("Status", STATUS_OPTIONS, key="queue_status")

    visible_cases = _active_filtered_cases()
    ordered_ids = ordered_case_ids(visible_cases)
    cases_by_id = {case.case_id: case for case in visible_cases}

    if not ordered_ids:
        st.info("No complaints match the current filters.")
        st.button(
            "Reset filters",
            key="reset_queue_filters",
            icon=":material/restart_alt:",
            on_click=_reset_queue_filters,
        )
        return

    st.markdown(
        "<div class='queue-heading'>Case and issue &nbsp;&nbsp;&nbsp;&nbsp; AI priority "
        "&nbsp;&nbsp;&nbsp;&nbsp; Status &nbsp;&nbsp;&nbsp;&nbsp; Assignee "
        "&nbsp;&nbsp;&nbsp;&nbsp; Received and response due</div>",
        unsafe_allow_html=True,
    )
    for case_id in ordered_ids:
        case = cases_by_id[case_id]
        details, priority_column, status_column, owner_column, time_column, action_column = (
            st.columns(QUEUE_COLUMN_WIDTHS)
        )
        with details:
            st.markdown(f"**{case.case_id}**  \n{case.issue_summary}")
        with priority_column:
            st.markdown(
                f"<span class='priority priority-{case.ai_priority.lower()}'>{case.ai_priority}</span>",
                unsafe_allow_html=True,
            )
        with status_column:
            st.markdown(case.status)
        with owner_column:
            st.markdown(case.assignee or "Unassigned")
        with time_column:
            st.caption(f"Received {_format_received(case)}")
            st.caption(f"Due {_format_due(case)}")
        with action_column:
            st.button(
                "Open",
                key=f"open_{case.case_id}",
                icon=":material/arrow_forward:",
                on_click=navigate,
                args=("case",),
                kwargs={"case_id": case.case_id},
            )
        st.divider()


def render_reports() -> None:
    cases = st.session_state["cases"]
    active_cases = [case for case in cases if case.status != "Resolved"]
    escalations = [case for case in cases if case.status == "Escalated"]
    assigned = [case for case in cases if case.assignee == CURRENT_OFFICER]

    st.header("Operations reports")
    st.caption("Current workload snapshot")
    active_metric, escalation_metric, assigned_metric = st.columns(3)
    active_metric.metric("Active complaints", len(active_cases))
    escalation_metric.metric("Escalations", len(escalations))
    assigned_metric.metric("Assigned to me", len(assigned))
    st.subheader("Status overview")
    for status in ("New", "In review", "Escalated", "Resolved"):
        st.write(f"{status}: {sum(case.status == status for case in cases)}")


def main() -> None:
    st.set_page_config(
        page_title="Complaint operations",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_state()
    load_product_css()
    render_sidebar()
    if st.session_state["screen"] == "reports":
        render_reports()
    elif st.session_state["screen"] == "case":
        render_case_workspace()
    elif st.session_state["screen"] == "new":
        render_new_complaint()
    else:
        render_queue()


if __name__ == "__main__":
    main()
