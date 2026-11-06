"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

try:
    from .extraction import extract_answer, parse_steps
    from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn
except ImportError:
    from src.extraction import extract_answer, parse_steps
    from src.utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    q_text = question.strip() if isinstance(question, str) else str(question)
    return (
        "Solve the following math word problem step by step.\n"
        "You must strictly follow this exact output format:\n"
        "1. Write each deduction or calculation as a separate numbered step starting with "
        "'Step 1:', 'Step 2:', 'Step 3:', etc. on its own line.\n"
        "2. Do NOT use markdown headings (###), bullet points, LaTeX formatting, or \\boxed{}.\n"
        "3. Conclude your solution on a new final line in the exact format:\n"
        "Final Answer: <integer>\n"
        "Write only the plain integer after 'Final Answer:' (no units, commas, or currency symbols).\n\n"
        "Example format:\n"
        "Step 1: Compute the initial amount: 3 + 4 = 7.\n"
        "Step 2: Multiply by 2 to find the total: 7 * 2 = 14.\n"
        "Final Answer: 14\n\n"
        f"Question: {q_text}\n"
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    if not answers:
        return None

    counts: dict[int, int] = {}
    first_index: dict[int, int] = {}

    for idx, ans in enumerate(answers):
        if ans is None:
            continue
        if ans not in counts:
            counts[ans] = 0
            first_index[ans] = idx
        counts[ans] += 1

    if not counts:
        return None

    # Maximize (frequency, -first_occurrence_index)
    best_ans = max(counts.keys(), key=lambda a: (counts[a], -first_index[a]))
    return best_ans


def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    t0 = time.perf_counter()
    prompt = format_step_prompt(question)

    if generate_fn is None:
        if model is None or tokenizer is None:
            raise ValueError("Either generate_fn or (model, tokenizer) must be provided to run_greedy.")
        generate_fn = make_hf_generate_fn(model, tokenizer)

    gen_kwargs = dict(GREEDY_GEN_KWARGS)
    outputs = generate_fn(prompt, 1, gen_kwargs)

    if outputs:
        completion = outputs[0].get("text", "")
        generated_tokens = int(outputs[0].get("num_tokens", 0))
    else:
        completion = ""
        generated_tokens = 0

    steps = parse_steps(completion)
    answer = extract_answer(completion)
    elapsed = time.perf_counter() - t0

    return {
        "prompt": prompt,
        "completion": completion,
        "steps": steps,
        "answer": answer,
        "generation_calls": 1,
        "generated_tokens": generated_tokens,
        "time_seconds": float(elapsed),
    }


def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    t0 = time.perf_counter()
    if n <= 0:
        return {
            "candidates": [],
            "answer": None,
            "generation_calls": 0,
            "generated_tokens": 0,
            "time_seconds": float(time.perf_counter() - t0),
        }

    prompt = format_step_prompt(question)

    if generate_fn is None:
        if model is None or tokenizer is None:
            raise ValueError("Either generate_fn or (model, tokenizer) must be provided to run_self_consistency.")
        generate_fn = make_hf_generate_fn(model, tokenizer)

    gen_kwargs = dict(SAMPLING_GEN_KWARGS)
    gen_kwargs["temperature"] = temperature

    outputs = generate_fn(prompt, n, gen_kwargs)

    candidates: list[dict[str, Any]] = []
    for item in outputs:
        comp_text = item.get("text", "")
        num_toks = int(item.get("num_tokens", 0))
        steps = parse_steps(comp_text)
        ans = extract_answer(comp_text)
        candidates.append({
            "completion": comp_text,
            "steps": steps,
            "answer": ans,
            "num_tokens": num_toks,
        })

    voted_answer = compute_majority_vote([c["answer"] for c in candidates])
    total_tokens = sum(c["num_tokens"] for c in candidates)
    elapsed = time.perf_counter() - t0

    return {
        "candidates": candidates,
        "answer": voted_answer,
        "generation_calls": len(candidates),
        "generated_tokens": total_tokens,
        "time_seconds": float(elapsed),
    }
