import hashlib
import logging
from pathlib import Path

from pydantic import ValidationError
from utils.helpers import extract_json
from utils.llm_client import RequestMessage, gemma

from models import ReviewDecision, ReviewIssue, ReviewResult, TestOutcome

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str | None = None


def load_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = (
            Path(__file__).parent.parent / "prompts" / "reviewer.txt"
        ).read_text()
    return _SYSTEM_PROMPT


def enforce_guardrails(review: ReviewResult, outcome: TestOutcome) -> ReviewResult:
    """
    Algorithmic override: if tests failed, the decision cannot be APPROVE.
    Constructs new ReviewIssue instances rather than mutating the existing list.
    """
    if not outcome.failures:
        return review

    if review.decision == ReviewDecision.APPROVE:
        logger.warning(
            "Guardrail triggered: LLM approved with failing tests. "
            "Overriding to REVISE."
        )

    # Always ensure every failure has a corresponding blocking issue,
    # even if the LLM already decided REVISE but omitted some failures.
    existing_descriptions = {i.description for i in review.issues}
    new_issues = list(review.issues)

    for failure in outcome.failures:
        description = f"Test failed: {failure.name} — {failure.message}"
        if description not in existing_descriptions:
            new_issues.append(
                ReviewIssue(
                    id=issue_id_from_description(description),
                    description=description,
                    severity="blocking",
                )
            )

    return review.model_copy(
        update={
            "decision": ReviewDecision.REVISE,
            "issues": new_issues,
        }
    )


def issue_id_from_description(description: str) -> str:
    digest = hashlib.sha256(description.encode("utf-8")).hexdigest()
    return digest[:12]


def build_correction_prompt(raw_response: str, error: Exception) -> str:
    return (
        f"Your previous response failed validation with this error:\n\n"
        f"{error}\n\n"
        f"Your previous response was:\n\n"
        f"{raw_response}\n\n"
        f"Return ONLY the corrected raw JSON object that matches the "
        f"ReviewResult schema. No markdown fences. No explanation."
    )


async def run_reviewer(
    reviewer_context: str,
    outcome: TestOutcome,
    iteration: int = 0,
    max_retries: int = 2,
) -> ReviewResult:
    """
    Critiques code and test results, returning a strict ReviewResult.

    Args:
        reviewer_context:   Pre-built prompt string from build_reviewer_context().
        outcome:            Structured test results from the sandbox executor.
        iteration:          Current loop iteration (0-indexed), used for leniency.
        max_retries:        How many self-correction attempts before raising.

    Returns:
        A validated ReviewResult with guardrails applied.

    Raises:
        RuntimeError: If a valid ReviewResult cannot be produced after all retries.
    """
    logger.info(f"Reviewer Agent starting — iteration {iteration}")
    system_prompt = load_prompt()

    messages: list[RequestMessage] = [{"role": "user", "content": reviewer_context}]
    raw_response = ""

    for attempt in range(max_retries + 1):
        try:
            raw_response = await gemma.generate(
                system_prompt=system_prompt,
                messages=messages,
            )
        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Reviewer Agent: LLM call failed after {max_retries + 1} "
                    f"attempts: {e}"
                ) from e
            continue

        json_str = extract_json(raw_response)
        if not json_str:
            logger.warning(f"Reviewer Agent: no JSON found on attempt {attempt + 1}.")
            if attempt < max_retries:
                messages += [
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "user",
                        "content": build_correction_prompt(
                            raw_response,
                            ValueError(
                                "Response contained no extractable JSON object."
                            ),
                        ),
                    },
                ]
            continue

        try:
            review = ReviewResult.model_validate_json(json_str)
            safe_review = enforce_guardrails(review, outcome)

            logger.info(
                f"Reviewer Agent decision: {safe_review.decision.value} "
                f"(score: {safe_review.score}, iteration: {iteration}, "
                f"attempt: {attempt + 1}/{max_retries + 1})"
            )
            return safe_review

        except (ValidationError, ValueError) as e:
            logger.warning(
                f"Reviewer Agent: schema validation failed on attempt "
                f"{attempt + 1}: {e}"
            )
            if attempt < max_retries:
                messages += [
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "user",
                        "content": build_correction_prompt(raw_response, e),
                    },
                ]

    logger.error(
        f"Reviewer Agent failed after {max_retries + 1} attempts. "
        f"Last raw response:\n{raw_response}"
    )
    raise RuntimeError(
        f"Reviewer Agent could not produce a valid ReviewResult after "
        f"{max_retries + 1} attempts."
    )
