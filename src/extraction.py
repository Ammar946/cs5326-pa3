"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = (
        r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)"
        r"(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final\s+Answer)|$)"
    )
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _parse_int_from_tail(tail: str) -> int | None:
    """
    Parses an integer value from the text immediately following a 'Final Answer' delimiter.
    Returns None if the value is non-integer (e.g. 18.5) or malformed.
    """
    if not tail or not isinstance(tail, str):
        return None

    cleaned_tail = tail.strip()
    if not cleaned_tail:
        return None

    # Allow optional leading markdown (*, `), LaTeX $/ \boxed{, sign, currency ($), commas, and decimal part
    m = re.match(
        r"^[\s*`\\$]*(?:boxed\{\s*)?[\s*`\\]*([+-])?\s*\$?\s*([+-])?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?",
        cleaned_tail,
    )
    if not m:
        return None

    sign1, sign2, int_part, dec_part = m.group(1), m.group(2), m.group(3), m.group(4)
    # Disallow conflicting double signs like '+-5' or '--5'
    if sign1 and sign2:
        return None

    # Ensure the matched number is not immediately followed by another decimal point or fraction slash + digit
    after_pos = m.end()
    rest = cleaned_tail[after_pos:]
    if re.match(r"^\s*[/\.]\s*\d", rest):
        return None

    # If a decimal part is present, it must be strictly zero (e.g. '.00' is valid, '.5' is None)
    if dec_part is not None:
        frac_digits = dec_part[1:]
        if not frac_digits or any(ch != "0" for ch in frac_digits):
            return None

    sign = -1 if (sign1 == "-" or sign2 == "-") else 1
    try:
        value = int(int_part.replace(",", ""))
        return sign * value
    except (ValueError, TypeError):
        return None


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    try:
        if not completion_text or not isinstance(completion_text, str):
            return None

        # Primary pattern: 'Final Answer' at start of string or start of line (colon optional)
        line_pattern = r"(?:^|\n)\s*[#*`\s]*final\s+answer\b[#*`\s]*[:=]?\s*([^\n\r]*)"
        matches = re.findall(line_pattern, completion_text, flags=re.IGNORECASE)

        if not matches:
            # Secondary pattern: inline 'Final Answer:' with explicit colon/equals
            inline_pattern = r"[#*`\s]*final\s+answer[#*`\s]*[:=]\s*([^\n\r]*)"
            matches = re.findall(inline_pattern, completion_text, flags=re.IGNORECASE)

        if not matches:
            return None

        # Always inspect the LAST 'Final Answer' occurrence
        last_tail = matches[-1]
        return _parse_int_from_tail(last_tail)
    except Exception:
        return None
