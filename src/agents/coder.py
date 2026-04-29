import logging
import re
from pathlib import Path

from utils.llm_client import _RequestMessage, gemma

from models import Subtask, TaskPlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str | None = None


def load_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = (
            Path(__file__).parent.parent / "prompts" / "coder.txt"
        ).read_text()
    return _SYSTEM_PROMPT


def extract_code(raw_text: str) -> str | None:
    """
    Extracts code from <code>...</code> delimiters.
    Falls back to stripping markdown fences if delimiters are absent.
    Returns None if no code can be extracted.
    """
    # Primary: XML-style delimiters (what we instruct the model to use)
    match = re.search(r"<code>\s*(.*?)\s*</code>", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback 1: markdown code fence with optional language tag
    match = re.search(
        r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Fallback 2: assume entire response is code if it looks like Python
    stripped = raw_text.strip()
    if stripped.startswith(("def ", "class ", "import ", "from ", "#")):
        return stripped

    return None


def build_user_prompt(
    plan: TaskPlan,
    subtask: Subtask,
    current_code: str = "",
    reviewer_feedback: list[str] | None = None,
) -> str:
    prompt = f"""
OVERARCHING GOAL:
{plan.goal}

YOUR SPECIFIC SUBTASK:
{subtask.description}

TARGET FILE:
{subtask.file_path}
""".strip()

    if current_code and reviewer_feedback:
        issues = "\n".join(f"- {issue}" for issue in reviewer_feedback)
        prompt += (
            "\n\nCURRENT CODE (needs revision):\n"
            "<code>\n"
            f"{current_code}\n"
            "</code>\n\n"
            "REVIEWER IDENTIFIED THESE SPECIFIC ISSUES — fix only these, do not "
            "refactor unrelated code:\n"
            f"{issues}\n"
        )
    elif current_code:
        prompt += (
            "\n\nCURRENT CODE (context only — do not modify unless your subtask "
            "requires it):\n"
            "<code>\n"
            f"{current_code}\n"
            "</code>\n"
        )
    return prompt


async def run_coder(
    plan: TaskPlan,
    subtask: Subtask,
    current_code: str = "",
    reviewer_feedback: list[str] | None = None,
    max_retries: int = 2,
) -> str:
    """
    Generates implementation code for a specific subtask.

    Args:
        plan:               The overarching TaskPlan for goal context.
        subtask:            The specific Subtask this agent is responsible for.
        current_code:       Existing code to revise (passed in on REVISE loops).
        reviewer_feedback:  Specific issues from the Reviewer agent (revision
                    loop only).
        max_retries:        How many self-correction attempts before raising.

    Returns:
        Raw Python code as a string.

    Raises:
        RuntimeError: If the agent fails to return extractable code after all retries.
    """
    logger.info(f"Coder Agent starting — subtask: {subtask.id}")
    system_prompt = load_prompt()
    user_prompt = build_user_prompt(plan, subtask, current_code, reviewer_feedback)

    # Conversation history for self-correction retries
    messages: list[_RequestMessage] = [{"role": "user", "content": user_prompt}]
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
                    f"Coder Agent: LLM call failed after {max_retries + 1} attempts "
                    f"for subtask '{subtask.id}': {e}"
                ) from e
            continue

        code = extract_code(raw_response)

        if code:
            logger.info(
                f"Coder Agent succeeded for subtask '{subtask.id}' "
                f"(attempt {attempt + 1}/{max_retries + 1}, "
                f"{len(code.splitlines())} lines)"
            )
            return code

        # Extraction failed — feed the error back for self-correction
        logger.warning(
            f"Coder Agent: could not extract code on attempt {attempt + 1}. "
            f"Requesting self-correction."
        )

        if attempt < max_retries:
            messages += [
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": (
                        "Your response did not contain extractable code. "
                        "You must wrap your code in <code> and </code> tags "
                        "with no other text outside them. "
                        "Please rewrite your response now."
                    ),
                },
            ]

    # Exhausted all attempts
    logger.error(
        f"Coder Agent failed for subtask '{subtask.id}' after "
        f"{max_retries + 1} attempts. Last raw response:\n{raw_response}"
    )
    raise RuntimeError(
        f"Coder Agent could not extract code for subtask '{subtask.id}' "
        f"after {max_retries + 1} attempts."
    )
