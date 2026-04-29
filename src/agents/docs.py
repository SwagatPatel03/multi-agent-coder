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
            Path(__file__).parent.parent / "prompts" / "docs.txt"
        ).read_text()
    return _SYSTEM_PROMPT


def extract_content(raw_text: str) -> str | None:
    """
    Extracts content from <content>...</content> delimiters.
    Falls back to markdown fence extraction, then to bare text
    if the response looks like code or markdown.
    Returns None only if no usable content can be found.
    """
    # Primary: XML-style delimiters
    match = re.search(
        r"<content>\s*(.*?)\s*</content>",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback 1: markdown code fence
    match = re.search(
        r"```(?:python|markdown|md)?\s*(.*?)\s*```",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback 2: bare response that looks like code or markdown
    stripped = raw_text.strip()
    if stripped.startswith(("def ", "class ", "import ", "from ", "#", "##", "**")):
        return stripped

    return None


def build_user_prompt(
    plan: TaskPlan,
    subtask: Subtask,
    current_code: str = "",
    reviewer_feedback: list[str] | None = None,
) -> str:
    # Label changes based on whether target is source code or markdown
    is_markdown = subtask.file_path.endswith((".md", ".rst", ".txt"))
    content_label = (
        "EXISTING MARKDOWN" if is_markdown else "IMPLEMENTATION CODE TO DOCUMENT"
    )

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
            f"\n\n{content_label} (needs revision):\n"
            "<content>\n"
            f"{current_code}\n"
            "</content>\n\n"
            "REVIEWER IDENTIFIED THESE SPECIFIC ISSUES — "
            "fix only these, do not touch unrelated documentation:\n"
            f"{issues}\n"
        )
    elif current_code:
        prompt += (
            f"\n\n{content_label}:\n" "<content>\n" f"{current_code}\n" "</content>\n"
        )

    return prompt


async def run_docs(
    plan: TaskPlan,
    subtask: Subtask,
    current_code: str = "",
    reviewer_feedback: list[str] | None = None,
    max_retries: int = 2,
) -> str:
    """
    Generates documentation or docstrings for a specific subtask.

    Args:
        plan:               The overarching TaskPlan for goal context.
        subtask:            The specific Subtask this agent is responsible for.
        current_code:       The implementation code or existing file to document.
        reviewer_feedback:  Specific issues from the
                            Reviewer agent (revision loop only).
        max_retries:        How many self-correction attempts before raising.

    Returns:
        The fully documented code or markdown content as a string.

    Raises:
        RuntimeError: If extractable content cannot be produced after all retries.
    """
    logger.info(f"Docs Agent starting — subtask: {subtask.id}")
    system_prompt = load_prompt()
    user_prompt = build_user_prompt(plan, subtask, current_code, reviewer_feedback)

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
                    f"Docs Agent: LLM call failed after {max_retries + 1} "
                    f"attempts for subtask '{subtask.id}': {e}"
                ) from e
            continue

        content = extract_content(raw_response)

        if content:
            logger.info(
                f"Docs Agent succeeded for subtask '{subtask.id}' "
                f"(attempt {attempt + 1}/{max_retries + 1}, "
                f"{len(content.splitlines())} lines)"
            )
            return content

        logger.warning(
            f"Docs Agent: could not extract content on attempt {attempt + 1}. "
            f"Requesting self-correction."
        )

        if attempt < max_retries:
            messages += [
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": (
                        "Your response did not contain extractable content. "
                        "You must wrap your entire output in <content> and </content> "
                        "tags with no other text outside them. "
                        "Please rewrite your response now."
                    ),
                },
            ]

    logger.error(
        f"Docs Agent failed for subtask '{subtask.id}' after "
        f"{max_retries + 1} attempts. Last raw response:\n{raw_response}"
    )
    raise RuntimeError(
        f"Docs Agent could not extract content for subtask '{subtask.id}' "
        f"after {max_retries + 1} attempts."
    )
