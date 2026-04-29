import re


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
