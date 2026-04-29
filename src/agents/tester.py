import logging
import re
from pathlib import Path

from utils.llm_client import RequestMessage, gemma

from models import Subtask, TaskPlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str | None = None


def load_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = (
            Path(__file__).parent.parent / "prompts" / "tester.txt"
        ).read_text()
    return _SYSTEM_PROMPT


def extract_code(raw_text: str) -> str | None:
    """
    Extracts test code from <code>...</code> delimiters.
    Falls back to markdown fence extraction, then to bare Python heuristic.
    Returns None only if no usable code can be found.
    """
    # Primary: XML-style delimiters
    match = re.search(
        r"<code>\s*(.*?)\s*</code>",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback 1: markdown code fence
    match = re.search(
        r"```(?:python)?\s*(.*?)\s*```",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback 2: bare response that looks like Python test code
    stripped = raw_text.strip()
    if stripped.startswith(("import ", "from ", "def test_", "@pytest")):
        return stripped

    return None


def build_user_prompt(
    plan: TaskPlan,
    subtask: Subtask,
    implementation_code: str,
    current_tests: str = "",
    reviewer_feedback: list[str] | None = None,
) -> str:
    prompt = f"""
OVERARCHING GOAL:
{plan.goal}

YOUR SPECIFIC SUBTASK:
{subtask.description}

TARGET FILE:
{subtask.file_path}

IMPLEMENTATION CODE TO TEST:
<code>
{implementation_code}
</code>
""".strip()

    if current_tests and reviewer_feedback:
        issues = "\n".join(f"- {issue}" for issue in reviewer_feedback)
        prompt += (
            "\n\nCURRENT TEST CODE (needs revision):\n"
            "<code>\n"
            f"{current_tests}\n"
            "</code>\n\n"
            "REVIEWER IDENTIFIED THESE SPECIFIC ISSUES — "
            "fix only these, do not refactor passing tests:\n"
            f"{issues}\n"
        )
    elif current_tests:
        prompt += (
            "\n\nCURRENT TEST CODE (context only):\n"
            "<code>\n"
            f"{current_tests}\n"
            "</code>\n"
        )

    return prompt


async def run_tester(
    plan: TaskPlan,
    subtask: Subtask,
    implementation_code: str,
    current_tests: str = "",
    reviewer_feedback: list[str] | None = None,
    max_retries: int = 2,
) -> str:
    """
    Generates pytest test code for a specific subtask.

    Args:
        plan:                The overarching TaskPlan for goal context.
        subtask:             The specific Subtask this agent is responsible for.
        implementation_code: The coder agent's output — what is being tested.
        current_tests:       Existing test code to revise (revision loop only).
        reviewer_feedback:   Specific issues from the
                            Reviewer agent (revision loop only).
        max_retries:         How many self-correction attempts before raising.

    Returns:
        Raw pytest code as a string.

    Raises:
        RuntimeError: If extractable test code cannot be produced after all retries.
    """
    logger.info(f"Tester Agent starting — subtask: {subtask.id}")
    system_prompt = load_prompt()
    user_prompt = build_user_prompt(
        plan, subtask, implementation_code, current_tests, reviewer_feedback
    )

    messages: list[RequestMessage] = [{"role": "user", "content": user_prompt}]
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
                    f"Tester Agent: LLM call failed after {max_retries + 1} "
                    f"attempts for subtask '{subtask.id}': {e}"
                ) from e
            continue

        test_code = extract_code(raw_response)

        if test_code:
            logger.info(
                f"Tester Agent succeeded for subtask '{subtask.id}' "
                f"(attempt {attempt + 1}/{max_retries + 1}, "
                f"{len(test_code.splitlines())} lines)"
            )
            return test_code

        logger.warning(
            f"Tester Agent: could not extract code on attempt {attempt + 1}. "
            f"Requesting self-correction."
        )

        if attempt < max_retries:
            messages += [
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": (
                        "Your response did not contain extractable code. "
                        "You must wrap your test code in <code> and </code> "
                        "tags with no other text outside them. "
                        "Please rewrite your response now."
                    ),
                },
            ]

    logger.error(
        f"Tester Agent failed for subtask '{subtask.id}' after "
        f"{max_retries + 1} attempts. Last raw response:\n{raw_response}"
    )
    raise RuntimeError(
        f"Tester Agent could not extract test code for subtask '{subtask.id}' "
        f"after {max_retries + 1} attempts."
    )
