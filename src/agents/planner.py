import logging
from pathlib import Path

from pydantic import ValidationError
from utils.helpers import extract_json
from utils.llm_client import RequestMessage, gemma

from models import TaskPlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str | None = None


def load_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = (
            Path(__file__).parent.parent / "prompts" / "planner.txt"
        ).read_text()
    return _SYSTEM_PROMPT


def build_user_prompt(user_request: str) -> str:
    return f"Task: {user_request}"


def build_correction_prompt(raw_response: str, error: Exception) -> str:
    return (
        f"Your previous response failed validation with this error:\n\n"
        f"{error}\n\n"
        f"Your previous response was:\n\n"
        f"{raw_response}\n\n"
        f"Return ONLY the corrected raw JSON object that matches the TaskPlan schema. "
        f"No markdown fences. No explanation. No text before or after the JSON."
    )


async def run_planner(
    user_request: str,
    max_retries: int = 2,
) -> TaskPlan:
    """
    Takes a raw user request and returns a validated TaskPlan.

    Calls the LLM, extracts JSON from the response, and validates it
    against the TaskPlan Pydantic schema. On failure, feeds the error
    back to the model for self-correction up to max_retries times.

    Args:
        user_request:   The raw natural language task from the user.
        max_retries:    How many self-correction attempts before raising.

    Returns:
        A validated TaskPlan instance.

    Raises:
        RuntimeError: If a valid TaskPlan cannot be produced after all retries.
    """
    logger.info(f"Planner Agent starting — request: {user_request!r}")
    system_prompt = load_prompt()

    messages: list[RequestMessage] = [
        {"role": "user", "content": build_user_prompt(user_request)}
    ]
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
                    f"Planner Agent: LLM call failed after {max_retries + 1} "
                    f"attempts: {e}"
                ) from e
            continue

        # Step 1: extract JSON string from raw response
        json_str = extract_json(raw_response)
        if not json_str:
            extraction_error = ValueError(
                "Response contained no extractable JSON object. "
                "Ensure your entire response is a single JSON object "
                "with no surrounding text."
            )
            logger.warning(
                f"Planner Agent: no JSON found on attempt {attempt + 1}. "
                f"Requesting self-correction."
            )
            if attempt < max_retries:
                messages += [
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "user",
                        "content": build_correction_prompt(
                            raw_response, extraction_error
                        ),
                    },
                ]
            continue

        # Step 2: validate extracted JSON against TaskPlan schema
        try:
            plan = TaskPlan.model_validate_json(json_str)
            logger.info(
                f"Planner Agent succeeded — {len(plan.subtasks)} subtasks "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )
            return plan

        except (ValidationError, ValueError) as e:
            logger.warning(
                f"Planner Agent: schema validation failed on attempt "
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

    # Exhausted all attempts
    logger.error(
        f"Planner Agent failed after {max_retries + 1} attempts. "
        f"Last raw response:\n{raw_response}"
    )
    raise RuntimeError(
        f"Planner Agent could not produce a valid TaskPlan after "
        f"{max_retries + 1} attempts."
    )
