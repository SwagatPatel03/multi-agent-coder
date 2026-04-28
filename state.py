import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models import AgentOutputs, ReviewResult, TaskPlan, TestOutcome


class IterationRecord(BaseModel):
    iteration: int
    code_hash: str
    test_outcome: TestOutcome
    review: ReviewResult
    issues_raised: list[str]
    issues_resolved: list[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LoopState(BaseModel):
    plan: TaskPlan
    current_outputs: AgentOutputs
    history: list[IterationRecord] = Field(default_factory=list)
    resolved_issues: set[str] = Field(default_factory=set)
    unresolved_issues: set[str] = Field(default_factory=set)
    stale_issue_counts: dict[str, int] = Field(default_factory=dict)
    status: str = Field(
        default="running",
        description="'running', 'approved', 'rejected', or 'exhausted'",
    )


def init_loop_state(plan: TaskPlan, initial_outputs: AgentOutputs) -> LoopState:
    """Initializes the state object before the review loop begins."""
    return LoopState(plan=plan, current_outputs=initial_outputs)


def record_iteration(
    state: LoopState,
    outcome: TestOutcome,
    review: ReviewResult,
) -> LoopState:
    """Advances the state machine by one iteration."""
    prev_issues = state.unresolved_issues
    current_blocking = {i.id for i in review.issues if i.severity == "blocking"}

    resolved = prev_issues - current_blocking
    new_unresolved = current_blocking

    # Track how many times each issue has appeared without being fixed
    stale_counts = dict(state.stale_issue_counts)
    for issue in current_blocking:
        stale_counts[issue] = stale_counts.get(issue, 0) + 1

    # Reset counts for resolved issues
    for res in resolved:
        stale_counts.pop(res, None)

    code_hash = hashlib.sha256(state.current_outputs.code.encode()).hexdigest()[:8]

    record = IterationRecord(
        iteration=len(state.history),
        code_hash=code_hash,
        test_outcome=outcome,
        review=review,
        issues_raised=list(current_blocking),
        issues_resolved=list(resolved),
    )

    return state.model_copy(
        update={
            "history": state.history + [record],
            "resolved_issues": state.resolved_issues | resolved,
            "unresolved_issues": new_unresolved,
            "stale_issue_counts": stale_counts,
        }
    )


def is_noop_revision(state: LoopState) -> bool:
    """Detects if the coder failed to make any changes between iterations."""
    if len(state.history) < 2:
        return False
    current_hash = hashlib.sha256(state.current_outputs.code.encode()).hexdigest()[:8]
    prev_hash = state.history[-1].code_hash
    return current_hash == prev_hash


# def is_flipflopping(state: LoopState, window: int = 4) -> bool:
#     """Detects if the reviewer is stuck in an approve/revise loop."""
#     if len(state.history) < window:
#         return False
#     recent_decisions = [r.review.decision for r in state.history[-window:]]
#     unique_decisions = set(recent_decisions)
#     return all(decisions[i] != decisions[i - 1] for i in range(1, len(decisions)))


def is_flipflopping(state: LoopState, window: int = 5) -> bool:
    if len(state.history) < window:
        return False
    recent = state.history[-window:]
    decisions = [r.review.decision for r in recent]
    pass_rates = [r.test_outcome.passed / max(r.test_outcome.total, 1) for r in recent]
    flips = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i - 1])
    improving = all(
        pass_rates[i] >= pass_rates[i - 1] for i in range(1, len(pass_rates))
    )
    return flips >= (window - 1) and not improving
