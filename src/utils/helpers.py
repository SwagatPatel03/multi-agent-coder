import re


def extract_json(raw_text: str) -> str | None:
    """
    Extracts a JSON object from raw LLM output.
    Tries multiple strategies in order of reliability.
    Returns None if no JSON object can be found.
    """
    # Primary: JSON inside markdown fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: outermost { } pair
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start : end + 1].strip()

    return None
