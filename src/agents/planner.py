import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError
from utils.llm_client import _RequestMessage, gemma

from models import TaskPlan

logger = logging.getLogger(__name__)


def load_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "planner.txt"
    with open(prompt_path, "r") as f:
        return f.read()


def clean_json_response(raw_text: str) -> str:
    # Try to extract JSON from a markdown fence first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to finding the outermost { } pair
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        return raw_text[start : end + 1].strip()
    return raw_text.strip()


async def run_planner(user_request: str, max_retries: int = 2) -> TaskPlan:
    system_prompt = load_prompt()
    messages: list[_RequestMessage] = [
        {"role": "user", "content": f"Task: {user_request}"}
    ]

    for attempt in range(max_retries + 1):
        raw_response = await gemma.generate(
            system_prompt=system_prompt,
            messages=messages,
        )
        cleaned = clean_json_response(raw_response)

        try:
            return TaskPlan.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == max_retries:
                logger.error("Planner failed after %d attempts", max_retries + 1)
                raise RuntimeError(f"Planner produced invalid schema: {e}") from e

            # Feed the error back so the model can self-correct
            messages += [
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": (
                        "Your response failed schema validation: "
                        f"{str(e).splitlines()[0]}\n"
                        f"Return ONLY the corrected raw JSON, nothing else."
                    ),
                },
            ]

    raise RuntimeError("Planner failed without returning a plan")
