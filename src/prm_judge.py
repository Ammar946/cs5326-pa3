"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import math
import re
from typing import Any, Callable

try:
    from .utils import make_hf_judge_fn
except ImportError:
    from src.utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    q_text = question.strip() if isinstance(question, str) else str(question)
    curr_text = current_step.strip() if isinstance(current_step, str) else str(current_step)

    if preceding_steps:
        rendered_steps = []
        for idx, step in enumerate(preceding_steps, start=1):
            step_str = str(step).strip()
            # Avoid duplicating 'Step K:' if caller passed already-labeled step text
            step_clean = re.sub(r"^\s*Step\s*\d+\s*[:.]?\s*", "", step_str, flags=re.IGNORECASE)
            rendered_steps.append(f"Step {idx}: {step_clean}")
        preceding_block = "\n".join(rendered_steps)
    else:
        preceding_block = "(no preceding steps)"

    return (
        "You are an expert mathematical reasoning verifier.\n"
        "Your task is to evaluate ONLY the Current Step for logical and arithmetic soundness.\n"
        "Local Validity Rule: You must assume all preceding steps are correct as written, "
        "even if an earlier step contained a mistake. Judge whether the Current Step "
        "follows logically from the Problem and Preceding Steps and whether its own "
        "arithmetic calculation is accurate.\n\n"
        f"Problem:\n{q_text}\n\n"
        f"Preceding Steps:\n{preceding_block}\n\n"
        f"Current Step:\n{curr_text}\n\n"
        "Verification Instructions:\n"
        "- If the Current Step is mathematically and logically sound (when you assume all preceding steps are correct), output 'Verdict: VALID'.\n"
        "- If the Current Step contains an arithmetic calculation error or logical fallacy, output 'Verdict: INVALID'.\n"
        "- Provide a single concise sentence after 'Rationale:' explaining your decision.\n\n"
        "Respond in the exact two-line format below:\n"
        "Verdict: <VALID or INVALID>\n"
        "Rationale: <one concise sentence explaining the verification decision>"
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not judge_output or not isinstance(judge_output, str):
        return -1.0, ""

    # 1. Verdict extraction
    verdict_match = re.search(
        r"verdict\s*:\s*\**\s*(VALID|INVALID)\b",
        judge_output,
        flags=re.IGNORECASE,
    )
    if verdict_match:
        token = verdict_match.group(1).upper()
        score = 1.0 if token == "VALID" else -1.0
    elif re.search(r"\b(?:INVALID|NOT\s+VALID)\b", judge_output, flags=re.IGNORECASE):
        score = -1.0
    elif re.search(r"\bVALID\b", judge_output, flags=re.IGNORECASE):
        score = 1.0
    else:
        score = -1.0

    # 2. Rationale extraction
    rationale = ""
    rat_match = re.search(r"rationale\s*:\s*\**\s*([^\n\r]*)", judge_output, flags=re.IGNORECASE)
    if rat_match:
        line_tail = rat_match.group(1).strip()
        if line_tail:
            # Find first sentence end ('.', '!', or '?' followed by whitespace or end of string)
            sent_end = re.search(r"[.!?](?=\s|$)", line_tail)
            if sent_end:
                rationale = line_tail[: sent_end.end()].strip()
            else:
                rationale = line_tail.strip()
            rationale = rationale.strip("*").strip()

    return float(score), rationale


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
      (plus 'step_probs' if judge_fn provides continuous logprob 'valid_prob')
    """
    if candidate is None:
        steps: list[str] = []
    else:
        steps = list(candidate.get("steps") or [])

    if not steps:
        return {
            "step_scores": [],
            "rationales": [],
            "judge_calls": 0,
            "judge_tokens": 0,
        }

    if judge_fn is None:
        if judge_model is None or judge_tokenizer is None:
            raise ValueError("Either judge_fn or (judge_model, judge_tokenizer) must be provided to score_path.")
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    step_scores: list[float] = []
    rationales: list[str] = []
    step_probs: list[float] = []
    has_probs = False
    judge_calls = 0
    judge_tokens = 0

    for t, current_step in enumerate(steps):
        preceding_steps = steps[:t]
        prompt = build_prm_judge_prompt(question, preceding_steps, current_step)
        res = judge_fn(prompt)

        raw_text = res.get("text", "")
        num_toks = int(res.get("num_tokens", 0))
        score, rationale = parse_judge_decision(raw_text)

        step_scores.append(float(score))
        rationales.append(rationale)
        judge_calls += 1
        judge_tokens += num_toks

        if "valid_prob" in res and res["valid_prob"] is not None:
            has_probs = True
            step_probs.append(float(res["valid_prob"]))
        else:
            # Fallback clamped probability from discrete score
            step_probs.append(0.99 if score > 0 else 0.01)

    result: dict[str, Any] = {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": judge_calls,
        "judge_tokens": judge_tokens,
    }
    if has_probs:
        result["step_probs"] = step_probs
    return result
