"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import copy
import time
from typing import Any, Callable

try:
    from .generation import compute_majority_vote, run_self_consistency
    from .prm_judge import score_path
except ImportError:
    from src.generation import compute_majority_vote, run_self_consistency
    from src.prm_judge import score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0, or continuous probabilities for 'product_logprob')
    into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - 'product_logprob' (Optional Bonus): product of continuous step validity probabilities p_t in (0, 1).
      - An unknown strategy raises ValueError.
    """
    valid_strategies = {"min", "product", "step_penalty", "product_logprob"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown aggregation strategy: '{strategy}'. Must be one of {sorted(valid_strategies)}.")

    if not step_scores:
        return -1.0

    if strategy == "min":
        return float(min(step_scores))

    if strategy == "product":
        prod = 1.0
        for r in step_scores:
            p_t = min(0.99, max(0.01, (float(r) + 1.0) / 2.0))
            prod *= p_t
        return float(prod)

    if strategy == "step_penalty":
        n_steps = len(step_scores)
        mean_score = sum(float(r) for r in step_scores) / n_steps
        num_invalid = sum(1 for r in step_scores if float(r) == -1.0)
        return float(mean_score - float(penalty_lambda) * num_invalid)

    if strategy == "product_logprob":
        prod = 1.0
        for s in step_scores:
            val = float(s)
            # If caller passed discrete {-1.0, +1.0} instead of continuous probabilities in [0, 1]
            if val < 0.0:
                p_t = min(0.9999, max(1e-4, (val + 1.0) / 2.0))
            else:
                p_t = min(0.9999, max(1e-4, val))
            prod *= p_t
        return float(prod)

    raise ValueError(f"Unknown aggregation strategy: '{strategy}'")


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    if not candidate_paths:
        raise ValueError("candidate_paths must not be empty.")

    # 1. Compute path_score for each candidate
    path_scores: list[float] = []
    for cand in candidate_paths:
        steps = cand.get("steps")
        if steps is not None and len(steps) == 0:
            path_scores.append(-1.0)
            continue

        if aggregation_strategy == "product_logprob" and cand.get("step_probs"):
            raw_scores = list(cand.get("step_probs") or [])
        else:
            raw_scores = list(cand.get("step_scores") or [])

        if not raw_scores:
            path_scores.append(-1.0)
        else:
            score = aggregate_path_score(
                raw_scores,
                strategy=aggregation_strategy,
                penalty_lambda=penalty_lambda,
            )
            path_scores.append(float(score))

    # 2. Eligibility: exclude answer=None if at least one candidate has non-None answer
    non_none_indices = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if non_none_indices:
        eligible_indices = non_none_indices
    else:
        eligible_indices = list(range(len(candidate_paths)))

    # 3. Maximum path_score among eligible candidates (tolerance 1e-9)
    max_score = max(path_scores[i] for i in eligible_indices)
    tied_indices = [i for i in eligible_indices if abs(path_scores[i] - max_score) <= 1e-9]
    num_tied = len(tied_indices)

    # 4. Tie-breaking via majority vote over tied candidates' answers
    if num_tied == 1:
        chosen_idx = tied_indices[0]
    else:
        tied_answers = [candidate_paths[i].get("answer") for i in tied_indices]
        voted_ans = compute_majority_vote(tied_answers)
        if voted_ans is None:
            chosen_idx = tied_indices[0]
        else:
            chosen_idx = next(
                (i for i in tied_indices if candidate_paths[i].get("answer") == voted_ans),
                tied_indices[0],
            )

    # 5. Return a shallow copy of the chosen candidate dict with 'path_score' and 'num_tied'
    selected_copy = copy.copy(candidate_paths[chosen_idx])
    selected_copy["path_score"] = float(path_scores[chosen_idx])
    selected_copy["num_tied"] = int(num_tied)
    return selected_copy


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    t0 = time.perf_counter()

    if candidates is None:
        sc_res = run_self_consistency(
            model=model,
            tokenizer=gen_tokenizer,
            question=question,
            n=n,
            temperature=0.7,
            generate_fn=generate_fn,
        )
        cand_list = [copy.copy(c) for c in sc_res["candidates"]]
        generation_calls = int(sc_res["generation_calls"])
        generated_tokens = int(sc_res["generated_tokens"])
    else:
        cand_list = [copy.copy(c) for c in candidates]
        generation_calls = 0
        generated_tokens = 0

    total_judge_calls = 0
    total_judge_tokens = 0
    augmented_candidates: list[dict[str, Any]] = []

    for cand in cand_list:
        cand_copy = copy.copy(cand)
        scored = score_path(
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            question=question,
            candidate=cand_copy,
            judge_fn=judge_fn,
        )
        cand_copy["step_scores"] = scored["step_scores"]
        cand_copy["rationales"] = scored["rationales"]
        if "step_probs" in scored:
            cand_copy["step_probs"] = scored["step_probs"]

        total_judge_calls += int(scored["judge_calls"])
        total_judge_tokens += int(scored["judge_tokens"])
        augmented_candidates.append(cand_copy)

    selected = select_best_path(
        augmented_candidates,
        aggregation_strategy=aggregation_strategy,
        penalty_lambda=penalty_lambda,
    )
    elapsed = time.perf_counter() - t0

    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": augmented_candidates,
        "generation_calls": generation_calls,
        "judge_calls": total_judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": total_judge_tokens,
        "time_seconds": float(elapsed),
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    t0 = time.perf_counter()

    existing = [copy.copy(c) for c in existing_candidates] if existing_candidates else []
    deficit = max(0, target_calls - len(existing))

    if deficit > 0:
        new_sc = run_self_consistency(
            model=model,
            tokenizer=tokenizer,
            question=question,
            n=deficit,
            temperature=temperature,
            generate_fn=generate_fn,
        )
        new_cands = new_sc["candidates"]
        generation_calls = int(new_sc["generation_calls"])
        generated_tokens = int(new_sc["generated_tokens"])
    else:
        new_cands = []
        generation_calls = 0
        generated_tokens = 0

    combined_candidates = (existing + new_cands)[:target_calls]
    voted_answer = compute_majority_vote([c.get("answer") for c in combined_candidates])
    elapsed = time.perf_counter() - t0

    return {
        "candidates": combined_candidates,
        "answer": voted_answer,
        "generation_calls": generation_calls,
        "total_samples": len(combined_candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": float(elapsed),
    }
