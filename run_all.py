"""
run_all.py - End-to-end evaluation pipeline for CS 5326 Programming Assignment 3.

Features:
  - Evaluates all 6 required configurations + lambda sweep + optional bonus (product_logprob)
    across the 50-problem GSM8K benchmark (data/gsm8k_test_50.jsonl).
  - Enforces per-problem deterministic seeding: set_seed_for_problem(problem_index) -> 5326 + i.
  - Shares the 5 sampled paths across Self-Consistency (N=5) and all PRM Best-of-5 variants,
    scoring steps once with the 7B judge and reusing step scores across aggregation rules.
  - Extends the 5 shared paths with 10 additional samples for Compute-Matched SC (N=15)
    and computes the full Self-Consistency scaling curve for N = 1..15.
  - Incrementally checkpoints every completed problem to outputs/per_problem.jsonl so
    interrupted Kaggle sessions resume seamlessly without repeating work.
  - Automatically computes and saves:
      * outputs/results_summary.json
      * outputs/aggregation_ablation.json
      * outputs/sc_scaling_curve.json
      * outputs/per_problem.jsonl

Usage on Kaggle (GPU T4x2):
  python run_all.py
"""
import argparse
import copy
import json
import os
import time
from typing import Any

from src.extraction import extract_answer, parse_steps
from src.generation import (
    compute_majority_vote,
    format_step_prompt,
    run_greedy,
    run_self_consistency,
)
from src.prm_judge import score_path
from src.search import (
    aggregate_path_score,
    run_compute_matched_self_consistency,
    select_best_path,
)
from src.utils import (
    GLOBAL_SEED,
    load_models,
    make_hf_generate_fn,
    make_hf_judge_fn,
    save_aggregation_ablation,
    save_results_summary,
    save_sc_scaling_curve,
    set_seed_for_problem,
)


def load_benchmark(dataset_path: str = "data/gsm8k_test_50.jsonl") -> list[dict[str, Any]]:
    """Loads the 50-problem GSM8K evaluation benchmark."""
    problems: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def load_existing_checkpoints(per_problem_path: str) -> dict[str, dict[str, Any]]:
    """Loads already-evaluated problem records from outputs/per_problem.jsonl for resumption."""
    completed: dict[str, dict[str, Any]] = {}
    if not os.path.exists(per_problem_path):
        return completed
    with open(per_problem_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec and "candidates" in rec and len(rec["candidates"]) == 15:
                    completed[rec["id"]] = rec
            except Exception:
                continue
    return completed


def write_all_checkpoints(records: list[dict[str, Any]], per_problem_path: str) -> None:
    """Writes all completed problem records in canonical order to outputs/per_problem.jsonl."""
    os.makedirs(os.path.dirname(per_problem_path), exist_ok=True)
    tmp_path = per_problem_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp_path, per_problem_path)


def evaluate_single_problem(
    problem_index: int,
    problem: dict[str, Any],
    gen_fn: Any,
    judge_fn: Any,
) -> dict[str, Any]:
    """
    Runs all configurations on a single benchmark problem with shared candidate pool:
      1. Greedy (N=1)
      2. Self-Consistency (N=5)
      3. PRM Judge over the 5 shared candidates (cached across aggregations)
      4. PRM Best-of-5 across min, product, step_penalty (lambda in {0.0, 0.5, 2.0}), and product_logprob
      5. Compute-Matched Self-Consistency (N=15, sampling 10 additional candidates)
    """
    set_seed_for_problem(problem_index)
    prob_id = problem["id"]
    question = problem["question"]
    gt_answer = int(problem["ground_truth_answer"])

    # 1. Greedy Baseline (N=1)
    greedy_res = run_greedy(question=question, generate_fn=gen_fn)

    # 2. Self-Consistency (N=5) - Shared 5-Candidate Pool
    sc5_res = run_self_consistency(
        question=question,
        n=5,
        temperature=0.7,
        generate_fn=gen_fn,
    )
    shared_5 = [copy.deepcopy(c) for c in sc5_res["candidates"]]

    # 3. PRM Step-Level Verification over the 5 shared candidates (scored once, reused across rules)
    t_judge_start = time.perf_counter()
    judge_calls_total = 0
    judge_tokens_total = 0
    scored_5: list[dict[str, Any]] = []

    for cand in shared_5:
        cand_copy = copy.deepcopy(cand)
        s_res = score_path(
            question=question,
            candidate=cand_copy,
            judge_fn=judge_fn,
        )
        cand_copy["step_scores"] = s_res["step_scores"]
        cand_copy["rationales"] = s_res["rationales"]
        if "step_probs" in s_res:
            cand_copy["step_probs"] = s_res["step_probs"]
        judge_calls_total += int(s_res["judge_calls"])
        judge_tokens_total += int(s_res["judge_tokens"])
        scored_5.append(cand_copy)

    judge_time_s = float(time.perf_counter() - t_judge_start)

    # 4. PRM Best-of-5 Selections across aggregation strategies
    prm_selections: dict[str, dict[str, Any]] = {}
    agg_configs = [
        ("min", "min", 2.0),
        ("product", "product", 2.0),
        ("step_penalty_0.0", "step_penalty", 0.0),
        ("step_penalty_0.5", "step_penalty", 0.5),
        ("step_penalty_2.0", "step_penalty", 2.0),
        ("product_logprob", "product_logprob", 2.0),
    ]
    for key_name, strat, lam in agg_configs:
        t_sel = time.perf_counter()
        sel = select_best_path(scored_5, aggregation_strategy=strat, penalty_lambda=lam)
        sel_time = float(time.perf_counter() - t_sel)
        prm_selections[key_name] = {
            "answer": sel.get("answer"),
            "path_score": float(sel["path_score"]),
            "num_tied": int(sel["num_tied"]),
            "time_seconds": sel_time,
        }

    # 5. Compute-Matched Self-Consistency (N=15): retains scored_5 (indices 0..4) + samples 10 new (indices 5..14)
    sc15_res = run_compute_matched_self_consistency(
        question=question,
        target_calls=15,
        temperature=0.7,
        existing_candidates=scored_5,
        generate_fn=gen_fn,
    )
    all_15_candidates = sc15_res["candidates"]

    # Scaling curve answers for N = 1..15
    sc_curve_answers = [
        compute_majority_vote([c.get("answer") for c in all_15_candidates[:n_sub]])
        for n_sub in range(1, 16)
    ]
    sc_curve_tokens = [
        sum(int(c.get("num_tokens", 0)) for c in all_15_candidates[:n_sub])
        for n_sub in range(1, 16)
    ]

    record = {
        "id": prob_id,
        "problem_index": problem_index,
        "question": question,
        "ground_truth_answer": gt_answer,
        "greedy": {
            "completion": greedy_res["completion"],
            "steps": greedy_res["steps"],
            "answer": greedy_res["answer"],
            "correct": greedy_res["answer"] == gt_answer,
            "generation_calls": int(greedy_res["generation_calls"]),
            "generated_tokens": int(greedy_res["generated_tokens"]),
            "time_seconds": float(greedy_res["time_seconds"]),
        },
        "shared_candidates_5": scored_5,
        "candidates": all_15_candidates,
        "predictions": {
            "greedy": greedy_res["answer"],
            "sc_5": sc5_res["answer"],
            "sc_15": sc15_res["answer"],
            "prm_min": prm_selections["min"]["answer"],
            "prm_product": prm_selections["product"]["answer"],
            "prm_step_penalty_0.0": prm_selections["step_penalty_0.0"]["answer"],
            "prm_step_penalty_0.5": prm_selections["step_penalty_0.5"]["answer"],
            "prm_step_penalty_2.0": prm_selections["step_penalty_2.0"]["answer"],
            "prm_product_logprob": prm_selections["product_logprob"]["answer"],
        },
        "prm_selections": prm_selections,
        "sc_curve_answers": sc_curve_answers,
        "sc_curve_tokens": sc_curve_tokens,
        "bookkeeping": {
            "greedy_calls": int(greedy_res["generation_calls"]),
            "greedy_tokens": int(greedy_res["generated_tokens"]),
            "greedy_time_s": float(greedy_res["time_seconds"]),
            "sc5_calls": int(sc5_res["generation_calls"]),
            "sc5_tokens": int(sc5_res["generated_tokens"]),
            "sc5_time_s": float(sc5_res["time_seconds"]),
            "sc10_extra_calls": int(sc15_res["generation_calls"]),
            "sc10_extra_tokens": int(sc15_res["generated_tokens"]),
            "sc10_extra_time_s": float(sc15_res["time_seconds"]),
            "sc15_calls": int(sc5_res["generation_calls"]) + int(sc15_res["generation_calls"]),
            "sc15_tokens": int(sc5_res["generated_tokens"]) + int(sc15_res["generated_tokens"]),
            "sc15_time_s": float(sc5_res["time_seconds"]) + float(sc15_res["time_seconds"]),
            "judge_calls": judge_calls_total,
            "judge_tokens": judge_tokens_total,
            "judge_time_s": judge_time_s,
        },
    }
    return record


def compute_and_save_artifacts(
    records: list[dict[str, Any]],
    output_dir: str = "outputs",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Computes and saves results_summary.json, aggregation_ablation.json, and sc_scaling_curve.json
    from the per-problem evaluation records.
    """
    n_probs = len(records)
    if n_probs == 0:
        raise ValueError("No problem records available to summarize.")

    def acc_pct(pred_key: str) -> float:
        correct = sum(
            1 for r in records if r["predictions"].get(pred_key) == r["ground_truth_answer"]
        )
        return round(100.0 * correct / n_probs, 2)

    # Aggregate call, token, and time totals across all 50 problems
    greedy_gen_calls = sum(r["bookkeeping"]["greedy_calls"] for r in records)
    greedy_gen_tokens = sum(r["bookkeeping"]["greedy_tokens"] for r in records)
    greedy_time = sum(r["bookkeeping"]["greedy_time_s"] for r in records)

    sc5_gen_calls = sum(r["bookkeeping"]["sc5_calls"] for r in records)
    sc5_gen_tokens = sum(r["bookkeeping"]["sc5_tokens"] for r in records)
    sc5_time = sum(r["bookkeeping"]["sc5_time_s"] for r in records)

    sc15_gen_calls = sum(r["bookkeeping"]["sc15_calls"] for r in records)
    sc15_gen_tokens = sum(r["bookkeeping"]["sc15_tokens"] for r in records)
    sc15_time = sum(r["bookkeeping"]["sc15_time_s"] for r in records)

    judge_calls = sum(r["bookkeeping"]["judge_calls"] for r in records)
    judge_tokens = sum(r["bookkeeping"]["judge_tokens"] for r in records)
    judge_time = sum(r["bookkeeping"]["judge_time_s"] for r in records)

    prm_min_sel_time = sum(r["prm_selections"]["min"]["time_seconds"] for r in records)
    prm_prod_sel_time = sum(r["prm_selections"]["product"]["time_seconds"] for r in records)
    prm_pen2_sel_time = sum(r["prm_selections"]["step_penalty_2.0"]["time_seconds"] for r in records)

    # 1. results_summary.json (6 mandatory rows)
    results_summary = [
        {
            "Method": "Greedy (N=1)",
            "Accuracy (%)": acc_pct("greedy"),
            "Generation Calls": greedy_gen_calls,
            "Judge Calls": 0,
            "Total LLM Calls": greedy_gen_calls,
            "Avg Gen Tokens / Problem": round(greedy_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": 0.0,
            "Total Time (s)": round(greedy_time, 2),
        },
        {
            "Method": "Self-Consistency (N=5)",
            "Accuracy (%)": acc_pct("sc_5"),
            "Generation Calls": sc5_gen_calls,
            "Judge Calls": 0,
            "Total LLM Calls": sc5_gen_calls,
            "Avg Gen Tokens / Problem": round(sc5_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": 0.0,
            "Total Time (s)": round(sc5_time, 2),
        },
        {
            "Method": "Compute-Matched SC (N=15)",
            "Accuracy (%)": acc_pct("sc_15"),
            "Generation Calls": sc15_gen_calls,
            "Judge Calls": 0,
            "Total LLM Calls": sc15_gen_calls,
            "Avg Gen Tokens / Problem": round(sc15_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": 0.0,
            "Total Time (s)": round(sc15_time, 2),
        },
        {
            "Method": "PRM Best-of-5 (min)",
            "Accuracy (%)": acc_pct("prm_min"),
            "Generation Calls": sc5_gen_calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": sc5_gen_calls + judge_calls,
            "Avg Gen Tokens / Problem": round(sc5_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": round(judge_tokens / n_probs, 2),
            "Total Time (s)": round(sc5_time + judge_time + prm_min_sel_time, 2),
        },
        {
            "Method": "PRM Best-of-5 (product)",
            "Accuracy (%)": acc_pct("prm_product"),
            "Generation Calls": sc5_gen_calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": sc5_gen_calls + judge_calls,
            "Avg Gen Tokens / Problem": round(sc5_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": round(judge_tokens / n_probs, 2),
            "Total Time (s)": round(sc5_time + judge_time + prm_prod_sel_time, 2),
        },
        {
            "Method": "PRM Best-of-5 (step_penalty, lambda=2.0)",
            "Accuracy (%)": acc_pct("prm_step_penalty_2.0"),
            "Generation Calls": sc5_gen_calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": sc5_gen_calls + judge_calls,
            "Avg Gen Tokens / Problem": round(sc5_gen_tokens / n_probs, 2),
            "Avg Judge Tokens / Problem": round(judge_tokens / n_probs, 2),
            "Total Time (s)": round(sc5_time + judge_time + prm_pen2_sel_time, 2),
        },
    ]

    # 2. aggregation_ablation.json
    def tie_rate_pct(sel_key: str) -> float:
        ties = sum(1 for r in records if int(r["prm_selections"][sel_key]["num_tied"]) > 1)
        return round(100.0 * ties / n_probs, 2)

    aggregation_ablation = [
        {
            "Strategy": "min",
            "Lambda": None,
            "Accuracy (%)": acc_pct("prm_min"),
            "Tie Rate (%)": tie_rate_pct("min"),
        },
        {
            "Strategy": "product",
            "Lambda": None,
            "Accuracy (%)": acc_pct("prm_product"),
            "Tie Rate (%)": tie_rate_pct("product"),
        },
        {
            "Strategy": "step_penalty",
            "Lambda": 0.0,
            "Accuracy (%)": acc_pct("prm_step_penalty_0.0"),
            "Tie Rate (%)": tie_rate_pct("step_penalty_0.0"),
        },
        {
            "Strategy": "step_penalty",
            "Lambda": 0.5,
            "Accuracy (%)": acc_pct("prm_step_penalty_0.5"),
            "Tie Rate (%)": tie_rate_pct("step_penalty_0.5"),
        },
        {
            "Strategy": "step_penalty",
            "Lambda": 2.0,
            "Accuracy (%)": acc_pct("prm_step_penalty_2.0"),
            "Tie Rate (%)": tie_rate_pct("step_penalty_2.0"),
        },
    ]

    # Save bonus product_logprob stats alongside in outputs/bonus_product_logprob.json
    bonus_stats = {
        "Strategy": "product_logprob",
        "Lambda": None,
        "Accuracy (%)": acc_pct("prm_product_logprob"),
        "Tie Rate (%)": tie_rate_pct("product_logprob"),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "bonus_product_logprob.json"), "w", encoding="utf-8") as f:
        json.dump(bonus_stats, f, indent=2)

    # 3. sc_scaling_curve.json (N = 1..15)
    sc_scaling_curve = []
    for n_sub in range(1, 16):
        idx = n_sub - 1
        correct_n = sum(
            1
            for r in records
            if compute_majority_vote([c.get("answer") for c in r["candidates"][:n_sub]])
            == r["ground_truth_answer"]
        )
        tokens_n = sum(
            sum(int(c.get("num_tokens", 0)) for c in r["candidates"][:n_sub])
            for r in records
        )
        sc_scaling_curve.append({
            "N": n_sub,
            "Accuracy (%)": round(100.0 * correct_n / n_probs, 2),
            "Avg Gen Tokens / Problem": round(tokens_n / n_probs, 2),
        })

    save_results_summary(results_summary, os.path.join(output_dir, "results_summary.json"))
    save_aggregation_ablation(aggregation_ablation, os.path.join(output_dir, "aggregation_ablation.json"))
    save_sc_scaling_curve(sc_scaling_curve, os.path.join(output_dir, "sc_scaling_curve.json"))

    return results_summary, aggregation_ablation, sc_scaling_curve


def main():
    parser = argparse.ArgumentParser(description="Run CS 5326 PA3 GSM8K Benchmark Evaluation")
    parser.add_argument("--dataset", type=str, default="data/gsm8k_test_50.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--gen-device", type=str, default="cuda:0")
    parser.add_argument("--judge-device", type=str, default="cuda:1")
    parser.add_argument("--mock", action="store_true", help="Run offline with synthetic mock callables for pipeline testing")
    args = parser.parse_args()

    problems = load_benchmark(args.dataset)
    per_problem_path = os.path.join(args.output_dir, "per_problem.jsonl")
    completed_map = load_existing_checkpoints(per_problem_path)
    print(f"Loaded {len(problems)} problems from {args.dataset}. Found {len(completed_map)} existing checkpoints.")

    if args.mock:
        print("Running in offline --mock mode...")
        def mock_gen_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
            return [
                {
                    "text": "Step 1: Compute intermediate value 9 * 2 = 18.\nFinal Answer: 18",
                    "num_tokens": 16,
                }
                for _ in range(num_samples)
            ]

        def mock_judge_fn(judge_prompt: str) -> dict:
            return {
                "text": "Verdict: VALID\nRationale: The arithmetic and logic are sound.",
                "num_tokens": 12,
                "valid_prob": 0.96,
            }

        gen_fn = mock_gen_fn
        judge_fn = mock_judge_fn
    else:
        if len(completed_map) < len(problems):
            print(f"Loading Generator on {args.gen_device} and PRM Judge on {args.judge_device}...")
            gen_model, gen_tokenizer, judge_model, judge_tokenizer = load_models(
                generator_device=args.gen_device,
                judge_device=args.judge_device,
            )
            gen_fn = make_hf_generate_fn(gen_model, gen_tokenizer)
            judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)
        else:
            gen_fn = None
            judge_fn = None

    ordered_records: list[dict[str, Any]] = []
    for idx, prob in enumerate(problems):
        prob_id = prob["id"]
        if prob_id in completed_map:
            rec = completed_map[prob_id]
            ordered_records.append(rec)
            print(f"[{idx + 1:02d}/{len(problems):02d}] {prob_id} (restored from checkpoint)")
            continue

        t0 = time.perf_counter()
        rec = evaluate_single_problem(
            problem_index=idx,
            problem=prob,
            gen_fn=gen_fn,
            judge_fn=judge_fn,
        )
        ordered_records.append(rec)
        completed_map[prob_id] = rec

        # Save incremental checkpoint immediately
        write_all_checkpoints(ordered_records, per_problem_path)
        dt = time.perf_counter() - t0
        preds = rec["predictions"]
        gt = rec["ground_truth_answer"]
        print(
            f"[{idx + 1:02d}/{len(problems):02d}] {prob_id} | GT={gt} | "
            f"Greedy={preds['greedy']} | SC@5={preds['sc_5']} | "
            f"PRM(min)={preds['prm_min']} | SC@15={preds['sc_15']} | ({dt:.1f}s)"
        )

    # Ensure final canonical order in per_problem.jsonl
    write_all_checkpoints(ordered_records, per_problem_path)
    summary, ablation, curve = compute_and_save_artifacts(ordered_records, args.output_dir)

    print("\n=== Benchmark Summary (outputs/results_summary.json) ===")
    for row in summary:
        print(
            f"  {row['Method']:<40} | Acc: {row['Accuracy (%)']:>6.2f}% | "
            f"Calls: {row['Total LLM Calls']:>4d} | Time: {row['Total Time (s)']:>7.2f}s"
        )

    print("\n=== Aggregation Ablation (outputs/aggregation_ablation.json) ===")
    for row in ablation:
        print(
            f"  Strategy: {row['Strategy']:<15} | Lambda: {str(row['Lambda']):<5} | "
            f"Acc: {row['Accuracy (%)']:>6.2f}% | Tie Rate: {row['Tie Rate (%)']:>6.2f}%"
        )


if __name__ == "__main__":
    main()

