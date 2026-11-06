"""
tests/test_public_adapter_smoke.py - Public smoke test suite for submission_adapter.py.

This test suite runs completely offline with no network, GPU, or third-party dependencies.
It validates that your implementation conforms strictly to the submission_adapter contract.

Usage:
  pytest -v tests/test_public_adapter_smoke.py
  python tests/test_public_adapter_smoke.py
"""
import copy
import os
import sys

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import submission_adapter as adapter


# =====================================================================
# Mock Helpers for Autograding Verification
# =====================================================================
def make_mock_generate_fn(responses: list[str], token_counts: list[int] | None = None):
    """Factory creating a mock GenerateFn for offline testing."""
    call_idx = [0]
    recorded_kwargs = []

    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        recorded_kwargs.append(copy.deepcopy(gen_kwargs))
        results = []
        for _ in range(num_samples):
            idx = call_idx[0] % len(responses)
            call_idx[0] += 1
            text = responses[idx]
            toks = token_counts[idx] if token_counts else len(text.split())
            results.append({"text": text, "num_tokens": toks})
        return results

    generate_fn.recorded_kwargs = recorded_kwargs  # type: ignore
    generate_fn.call_idx = call_idx  # type: ignore
    return generate_fn


def make_mock_judge_fn(decisions: list[str], token_counts: list[int] | None = None):
    """Factory creating a mock JudgeFn for offline testing."""
    call_idx = [0]
    prompts_seen = []

    def judge_fn(judge_prompt: str) -> dict:
        prompts_seen.append(judge_prompt)
        idx = call_idx[0] % len(decisions)
        call_idx[0] += 1
        text = decisions[idx]
        toks = token_counts[idx] if token_counts else len(text.split())
        return {"text": text, "num_tokens": toks}

    judge_fn.prompts_seen = prompts_seen  # type: ignore
    judge_fn.call_idx = call_idx  # type: ignore
    return judge_fn


# =====================================================================
# Task 1: Prompting, Extraction, and Baselines Tests
# =====================================================================
def test_format_step_prompt():
    prompt = adapter.format_step_prompt("What is 10 + 20?")
    assert isinstance(prompt, str)
    assert "What is 10 + 20?" in prompt
    assert "Step 1" in prompt
    assert "Final Answer" in prompt


def test_parse_steps_plain():
    text = (
        "Step 1: Compute 10 + 20 = 30.\n"
        "Step 2: Multiply 30 by 2 to get 60.\n"
        "Final Answer: 60"
    )
    steps = adapter.parse_steps(text)
    assert len(steps) == 2
    assert "Compute 10 + 20 = 30." in steps[0]
    assert "Multiply 30 by 2 to get 60." in steps[1]


def test_parse_steps_markdown_decorated():
    text = (
        "**Step 1:** Add 5 and 5 to get 10.\n"
        "### Step 2: Multiply 10 * 3.\n"
        "**Final Answer:** 30"
    )
    steps = adapter.parse_steps(text)
    assert len(steps) == 2
    assert "Add 5 and 5 to get 10." in steps[0]
    assert not steps[0].startswith("*")
    assert "Multiply 10 * 3." in steps[1]
    assert not steps[1].startswith("#")


def test_extract_answer_required_behaviors():
    # Exactly the 11 rows specified in the assignment manual
    cases = [
        ("Final Answer: 18", 18),
        ("Final Answer: $1,234", 1234),
        ("Final Answer: 18.00", 18),
        ("Final Answer: 18.5", None),
        ("Final Answer: -5", -5),
        ("Final Answer: 18 dollars.", 18),
        ("**Final Answer:** 18", 18),
        ("final answer 18", 18),
        ("First line.\nFinal Answer: 10\nSecond line.\nFinal Answer: 42", 42),
        ("There is no answer here.", None),
        ("", None),
    ]
    for completion, expected in cases:
        actual = adapter.extract_answer(completion)
        assert actual == expected, f"Failed for '{completion}': expected {expected}, got {actual}"


def test_extract_answer_resilience():
    assert adapter.extract_answer(None) is None  # type: ignore
    assert adapter.extract_answer("Final Answer: [object Object]") is None
    assert adapter.extract_answer("Final Answer: None") is None


def test_compute_majority_vote():
    # Unanimous
    assert adapter.compute_majority_vote([42, 42, 42]) == 42
    # Clear winner
    assert adapter.compute_majority_vote([10, 20, 10, 30]) == 10
    # Tie broken to earliest candidate
    assert adapter.compute_majority_vote([10, 20, 20, 10]) == 10
    assert adapter.compute_majority_vote([20, 10, 10, 20]) == 20
    # Ignores Nones
    assert adapter.compute_majority_vote([None, 15, None, 15, 7]) == 15
    # All None or empty
    assert adapter.compute_majority_vote([None, None]) is None
    assert adapter.compute_majority_vote([]) is None


def test_run_greedy():
    mock_gen = make_mock_generate_fn(
        ["Step 1: Add 2 and 3 to get 5.\nFinal Answer: 5"],
        token_counts=[12],
    )
    res = adapter.run_greedy(None, None, "What is 2 + 3?", generate_fn=mock_gen)
    assert res["answer"] == 5
    assert len(res["steps"]) == 1
    assert res["generation_calls"] == 1
    assert res["generated_tokens"] == 12
    assert "time_seconds" in res
    assert res["time_seconds"] >= 0.0
    assert "prompt" in res
    assert "completion" in res
    assert mock_gen.recorded_kwargs[0].get("do_sample") is False
    assert mock_gen.recorded_kwargs[0].get("temperature") is None
    assert mock_gen.recorded_kwargs[0].get("top_p") is None
    assert mock_gen.recorded_kwargs[0].get("top_k") is None


def test_run_self_consistency():
    responses = [
        "Step 1: Compute.\nFinal Answer: 10",
        "Step 1: Compute.\nFinal Answer: 20",
        "Step 1: Compute.\nFinal Answer: 10",
    ]
    mock_gen = make_mock_generate_fn(responses, token_counts=[10, 10, 10])
    res = adapter.run_self_consistency(None, None, "What is X?", n=3, temperature=0.7, generate_fn=mock_gen)
    assert res["answer"] == 10
    assert len(res["candidates"]) == 3
    assert res["generation_calls"] == 3
    assert res["generated_tokens"] == 30
    assert "time_seconds" in res
    assert res["time_seconds"] >= 0.0
    for cand in res["candidates"]:
        assert "completion" in cand
        assert "steps" in cand
        assert "answer" in cand
        assert "num_tokens" in cand
    assert mock_gen.recorded_kwargs[0].get("do_sample") is True
    assert mock_gen.recorded_kwargs[0].get("temperature") == 0.7


# =====================================================================
# Task 2: PRM Judge & Step Verification Tests
# =====================================================================
def test_build_prm_judge_prompt_structure():
    # Empty preceding steps
    p_first = adapter.build_prm_judge_prompt("What is 2 + 2?", [], "2 + 2 = 4")
    assert "What is 2 + 2?" in p_first
    assert "(no preceding steps)" in p_first
    assert "Current Step:" in p_first
    assert "2 + 2 = 4" in p_first
    assert "assume all preceding steps are correct" in p_first.lower()
    assert "Verdict: VALID" in p_first
    assert "Verdict: INVALID" in p_first
    assert "Rationale:" in p_first

    # Preceding steps included in order and numbered Step 1, Step 2
    preceding = ["Let x = 2", "Let y = 3"]
    p_later = adapter.build_prm_judge_prompt("Find x + y", preceding, "x + y = 5")
    assert "Step 1: Let x = 2" in p_later
    assert "Step 2: Let y = 3" in p_later
    assert p_later.index("Step 1: Let x = 2") < p_later.index("Step 2: Let y = 3")
    assert "Current Step:" in p_later
    assert "x + y = 5" in p_later
    assert "assume all preceding steps are correct" in p_later.lower()
    assert "Verdict: VALID" in p_later
    assert "Verdict: INVALID" in p_later
    assert "Rationale:" in p_later


def test_parse_judge_decision():
    cases = [
        ("Verdict: VALID\nRationale: Step is sound.", 1.0, "Step is sound."),
        ("Verdict: INVALID\nRationale: Calculation error.", -1.0, "Calculation error."),
        ("Verdict: **VALID**\nRationale: Correct algebra.", 1.0, "Correct algebra."),
        ("Verdict: **INVALID**\nRationale: Arithmetic flaw.", -1.0, "Arithmetic flaw."),
        ("verdict: invalid\nrationale: false premise.", -1.0, "false premise."),
        ("Verdict: VALID\nRationale: 2.5 * 4 = 10 is correct. Extra text.", 1.0, "2.5 * 4 = 10 is correct."),
        ("Verdict: VALID\nRationale: 2.5 \u00d7 4 = 10 is correct. Extra text.", 1.0, "2.5 \u00d7 4 = 10 is correct."),
        ("Verdict: INVALID\nRationale: Step is incorrect! Next explanation.", -1.0, "Step is incorrect!"),
        ("Verdict: INVALID\nRationale: Is this correct? No.", -1.0, "Is this correct?"),
        ("The step is NOT VALID because of bad arithmetic.", -1.0, ""),
        ("The step is valid and sound.", 1.0, ""),
        ("The step is invalid.", -1.0, ""),  # Substring trap: INVALID contains VALID
        ("Verdict: INVALID\nVerdict: VALID", -1.0, ""),  # First verdict line used
        ("Total gibberish output without decision.", -1.0, ""),
        ("", -1.0, ""),
    ]
    for judge_text, expected_score, expected_rat in cases:
        score, rationale = adapter.parse_judge_decision(judge_text)
        assert score == expected_score, f"Failed score for '{judge_text}': got {score}, expected {expected_score}"
        assert rationale == expected_rat, f"Failed rationale for '{judge_text}': got {repr(rationale)}, expected {repr(expected_rat)}"


def test_score_path():
    candidate = {
        "steps": ["Step 1: x = 10", "Step 2: y = 20", "Step 3: x + y = 30"]
    }
    judge_responses = [
        "Verdict: VALID\nRationale: Good x.",
        "Verdict: VALID\nRationale: Good y.",
        "Verdict: INVALID\nRationale: Bad sum.",
    ]
    mock_judge = make_mock_judge_fn(judge_responses, token_counts=[8, 8, 9])
    res = adapter.score_path(None, None, "Problem text", candidate, judge_fn=mock_judge)
    assert res["step_scores"] == [1.0, 1.0, -1.0]
    assert len(res["rationales"]) == 3
    assert res["judge_calls"] == 3
    assert res["judge_tokens"] == 25
    # Verify preceding steps were passed sequentially
    assert len(mock_judge.prompts_seen) == 3
    assert "Step 1: x = 10" in mock_judge.prompts_seen[1]
    assert "Step 2: y = 20" in mock_judge.prompts_seen[2]


def test_aggregate_path_score():
    # Empty list returns -1.0 for all strategies
    assert adapter.aggregate_path_score([]) == -1.0
    assert adapter.aggregate_path_score([], strategy="product") == -1.0
    assert adapter.aggregate_path_score([], strategy="step_penalty") == -1.0

    # min
    assert adapter.aggregate_path_score([1.0, 1.0, 1.0], strategy="min") == 1.0
    assert adapter.aggregate_path_score([1.0, -1.0, 1.0], strategy="min") == -1.0

    # product: p_t clamped in [0.01, 0.99]
    p_all = adapter.aggregate_path_score([1.0, 1.0, 1.0], strategy="product")
    assert abs(p_all - (0.99 ** 3)) < 1e-6
    p_one_inv = adapter.aggregate_path_score([1.0, -1.0, 1.0], strategy="product")
    assert abs(p_one_inv - (0.99 * 0.99 * 0.01)) < 1e-6

    # step_penalty: mean - lambda * num_invalid
    assert adapter.aggregate_path_score([1.0, 1.0], strategy="step_penalty", penalty_lambda=2.0) == 1.0
    assert adapter.aggregate_path_score([1.0, -1.0], strategy="step_penalty", penalty_lambda=2.0) == -2.0
    assert adapter.aggregate_path_score([1.0, -1.0], strategy="step_penalty", penalty_lambda=0.5) == -0.5

    # Unknown strategy raises ValueError
    try:
        adapter.aggregate_path_score([1.0], strategy="unsupported_strategy")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# =====================================================================
# Task 3: Best-of-N Search & Budget Matching Tests
# =====================================================================
def test_select_best_path_clear_winner():
    cands = [
        {"steps": ["s1"], "step_scores": [-1.0], "answer": 10},
        {"steps": ["s1"], "step_scores": [1.0], "answer": 20},
    ]
    orig_cands = copy.deepcopy(cands)
    win = adapter.select_best_path(cands, aggregation_strategy="min")
    assert win["answer"] == 20
    assert win["path_score"] == 1.0
    assert win["num_tied"] == 1
    # Verify inputs are not mutated
    assert cands == orig_cands

    # Verify candidate dict extra keys pass through unchanged in returned copy
    cands_extra = [
        {"steps": ["s1"], "step_scores": [1.0], "answer": 42, "custom_id": 999, "flag": True}
    ]
    win_extra = adapter.select_best_path(cands_extra)
    assert win_extra["answer"] == 42
    assert win_extra["custom_id"] == 999
    assert win_extra["flag"] is True
    assert win_extra["path_score"] == 1.0
    assert win_extra["num_tied"] == 1


def test_select_best_path_tie_broken_by_majority():
    # Candidates 0, 1, 2 all tie with score 1.0. Answers: 10, 20, 20.
    # Majority answer is 20. Candidate 1 is lowest-index candidate with answer 20.
    cands = [
        {"steps": ["s1"], "step_scores": [1.0], "answer": 10},
        {"steps": ["s1"], "step_scores": [1.0], "answer": 20},
        {"steps": ["s1"], "step_scores": [1.0], "answer": 20},
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="min")
    assert win["answer"] == 20
    assert win["num_tied"] == 3
    assert win["steps"] == ["s1"]


def test_select_best_path_answer_none_exclusion():
    cands = [
        {"steps": ["s1"], "step_scores": [1.0], "answer": None},  # Max score but answer is None
        {"steps": ["s1"], "step_scores": [-1.0], "answer": 42},   # Lower score but valid answer
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="min")
    assert win["answer"] == 42
    assert win["path_score"] == -1.0
    assert win["num_tied"] == 1


def test_select_best_path_all_answers_none():
    cands = [
        {"steps": ["s1"], "step_scores": [1.0], "answer": None},
        {"steps": ["s1"], "step_scores": [-1.0], "answer": None},
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="min")
    assert win["answer"] is None
    assert win["path_score"] == 1.0
    assert win["num_tied"] == 1


def test_run_prm_best_of_n_standalone():
    gen_texts = [
        "Step 1: Compute a.\nFinal Answer: 10",
        "Step 1: Compute b.\nFinal Answer: 20",
    ]
    mock_gen = make_mock_generate_fn(gen_texts, token_counts=[15, 15])
    judge_texts = [
        "Verdict: INVALID\nRationale: Math error.",
        "Verdict: VALID\nRationale: Correct logic.",
    ]
    mock_judge = make_mock_judge_fn(judge_texts, token_counts=[10, 10])

    res = adapter.run_prm_best_of_n(
        None, None, None, None, "What is X?", n=2, 
        generate_fn=mock_gen, judge_fn=mock_judge
    )
    assert res["answer"] == 20
    assert res["generation_calls"] == 2
    assert res["judge_calls"] == 2
    assert res["generated_tokens"] == 30
    assert res["judge_tokens"] == 20
    assert "time_seconds" in res
    assert res["time_seconds"] >= 0.0
    assert res["selected"]["path_score"] == 1.0
    assert len(res["candidates"]) == 2
    for cand in res["candidates"]:
        assert "step_scores" in cand
        assert "rationales" in cand


def test_run_prm_best_of_n_with_shared_candidates():
    shared_cands = [
        {"completion": "...", "steps": ["Step 1: a"], "answer": 10, "num_tokens": 12},
        {"completion": "...", "steps": ["Step 1: b"], "answer": 20, "num_tokens": 14},
    ]
    mock_gen = make_mock_generate_fn(["Should not be called"], token_counts=[100])
    judge_texts = [
        "Verdict: VALID\nRationale: All good.",
        "Verdict: INVALID\nRationale: Mistake.",
    ]
    mock_judge = make_mock_judge_fn(judge_texts, token_counts=[8, 8])

    res = adapter.run_prm_best_of_n(
        None, None, None, None, "Problem", n=2, 
        candidates=shared_cands, generate_fn=mock_gen, judge_fn=mock_judge
    )
    # Must NOT draw new samples
    assert res["generation_calls"] == 0
    assert res["generated_tokens"] == 0
    assert res["judge_calls"] == 2
    assert res["judge_tokens"] == 16
    assert "time_seconds" in res
    assert res["time_seconds"] >= 0.0
    assert res["answer"] == 10


def test_run_compute_matched_self_consistency():
    existing_cands = [
        {"completion": "...", "steps": ["s1"], "answer": 10, "num_tokens": 10},
        {"completion": "...", "steps": ["s1"], "answer": 20, "num_tokens": 10},
    ]
    new_responses = [
        "Step 1: ...\nFinal Answer: 20",
        "Step 1: ...\nFinal Answer: 20",
        "Step 1: ...\nFinal Answer: 30",
    ]
    mock_gen = make_mock_generate_fn(new_responses, token_counts=[12, 12, 12])

    res = adapter.run_compute_matched_self_consistency(
        None, None, "Problem", target_calls=5, 
        existing_candidates=existing_cands, generate_fn=mock_gen
    )
    assert res["total_samples"] == 5
    assert res["generation_calls"] == 3
    assert res["generated_tokens"] == 36
    assert len(res["candidates"]) == 5
    assert res["answer"] == 20
    assert "time_seconds" in res
    assert res["time_seconds"] >= 0.0


def test_scaling_curve_consistency():
    # 5 shared candidates (indices 0..4) and 10 additional candidates (indices 5..14)
    # Answers: indices 0..4 have majority 10; indices 0..14 have majority 20
    all_answers = [10, 10, 10, 20, 20] + [20, 20, 20, 20, 20, 20, 30, 30, 30, 30]
    sc5_ans = adapter.compute_majority_vote(all_answers[:5])
    assert sc5_ans == 10

    sc15_ans = adapter.compute_majority_vote(all_answers[:15])
    assert sc15_ans == 20

    # Consistency across curve: each point N is majority vote over the first N samples
    curve_answers = [adapter.compute_majority_vote(all_answers[:n]) for n in range(1, 16)]
    assert curve_answers[4] == sc5_ans    # N=5 (index 4) matches SC@5
    assert curve_answers[14] == sc15_ans  # N=15 (index 14) matches SC@15


def run_all_tests():
    """Runs all test functions without external dependencies."""
    test_funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    print(f"Running {len(test_funcs)} public adapter smoke tests...")
    for fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
    print("=" * 45)
    print(f"Test Summary: {passed} passed, {failed} failed.")
    if failed == 0:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
