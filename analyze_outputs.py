"""
analyze_outputs.py - Statistical analysis, bootstrap CIs, failure case mining,
judge audit extraction, and publication-quality plot generation for REPORT.pdf.

Usage:
  python3 analyze_outputs.py --output-dir outputs --fig-dir figures
"""
import argparse
import json
import os
import random
from typing import Any

from src.generation import compute_majority_vote


def load_per_problem(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_bootstrap_ci(
    correct_flags: list[int],
    n_resamples: int = 1000,
    seed: int = 5326,
) -> tuple[float, float, float]:
    """
    Computes mean accuracy (%) and 95% bootstrap confidence interval (2.5th, 97.5th percentiles)
    using 1,000 resamples over problems with seed 5326.
    """
    rng = random.Random(seed)
    n = len(correct_flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_acc = 100.0 * sum(correct_flags) / n
    boot_means = []
    for _ in range(n_resamples):
        sample = [correct_flags[rng.randrange(n)] for _ in range(n)]
        boot_means.append(100.0 * sum(sample) / n)
    boot_means.sort()
    low_idx = int(0.025 * n_resamples)
    high_idx = int(0.975 * n_resamples) - 1
    return round(mean_acc, 2), round(boot_means[low_idx], 2), round(boot_means[high_idx], 2)


def compute_paired_win_loss(
    records: list[dict[str, Any]],
    key_a: str,
    key_b: str,
) -> dict[str, int]:
    """
    Compares method A vs method B on each problem against ground_truth_answer:
      - a_wins: A is correct and B is wrong
      - b_wins: B is correct and A is wrong
      - both_correct: both A and B are correct
      - both_wrong: both A and B are wrong
    """
    a_wins = 0
    b_wins = 0
    both_correct = 0
    both_wrong = 0
    for r in records:
        gt = r["ground_truth_answer"]
        a_ok = r["predictions"].get(key_a) == gt
        b_ok = r["predictions"].get(key_b) == gt
        if a_ok and not b_ok:
            a_wins += 1
        elif b_ok and not a_ok:
            b_wins += 1
        elif a_ok and b_ok:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "both_agreed_correctness": both_correct + both_wrong,
    }


def generate_plots(
    results_summary: list[dict[str, Any]],
    scaling_curve: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    fig_dir: str,
) -> None:
    """Generates PDF plots for Section 1 and Section 2 of REPORT.pdf."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PDF plot generation.")
        return

    os.makedirs(fig_dir, exist_ok=True)

    # Extract SC curve arrays
    ns = [row["N"] for row in scaling_curve]
    sc_accs = [row["Accuracy (%)"] for row in scaling_curve]
    sc_toks = [row["Avg Gen Tokens / Problem"] for row in scaling_curve]
    sc_calls_total = [row["N"] * 50 for row in scaling_curve]

    summary_by_name = {row["Method"]: row for row in results_summary}
    greedy_row = summary_by_name["Greedy (N=1)"]
    prm_min_row = summary_by_name["PRM Best-of-5 (min)"]
    prm_prod_row = summary_by_name["PRM Best-of-5 (product)"]
    prm_pen_row = summary_by_name["PRM Best-of-5 (step_penalty, lambda=2.0)"]

    # Plot 1: Accuracy vs. Total LLM Calls
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        sc_calls_total,
        sc_accs,
        marker="o",
        linewidth=2.0,
        markersize=5,
        color="#1f77b4",
        label="Self-Consistency ($N=1\\dots 15$)",
    )
    ax.scatter(
        [greedy_row["Total LLM Calls"]],
        [greedy_row["Accuracy (%)"]],
        marker="D",
        s=75,
        color="#7f7f7f",
        zorder=5,
        label=f"Greedy ($N=1$): {greedy_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_min_row["Total LLM Calls"]],
        [prm_min_row["Accuracy (%)"]],
        marker="*",
        s=150,
        color="#d62728",
        zorder=6,
        label=f"PRM Best-of-5 (min): {prm_min_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_prod_row["Total LLM Calls"]],
        [prm_prod_row["Accuracy (%)"]],
        marker="^",
        s=90,
        color="#2ca02c",
        zorder=5,
        label=f"PRM Best-of-5 (product): {prm_prod_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_pen_row["Total LLM Calls"]],
        [prm_pen_row["Accuracy (%)"]],
        marker="s",
        s=80,
        color="#9467bd",
        zorder=5,
        label=f"PRM Best-of-5 (step_penalty): {prm_pen_row['Accuracy (%)']:.1f}%",
    )
    ax.set_xlabel("Total LLM Calls across 50 Problems (Gen + Judge)", fontsize=10)
    ax.set_ylabel("Exact-Match Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy vs. Total LLM Calls on GSM8K-50", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scaling_calls.pdf"))
    fig.savefig(os.path.join(fig_dir, "scaling_calls.png"), dpi=200)
    plt.close(fig)

    # Plot 2: Accuracy vs. Total Output Tokens per Problem
    prm_total_tok_per_prob = (
        prm_min_row["Avg Gen Tokens / Problem"] + prm_min_row["Avg Judge Tokens / Problem"]
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        sc_toks,
        sc_accs,
        marker="o",
        linewidth=2.0,
        markersize=5,
        color="#1f77b4",
        label="Self-Consistency ($N=1\\dots 15$)",
    )
    ax.scatter(
        [greedy_row["Avg Gen Tokens / Problem"]],
        [greedy_row["Accuracy (%)"]],
        marker="D",
        s=75,
        color="#7f7f7f",
        zorder=5,
        label=f"Greedy ($N=1$): {greedy_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_total_tok_per_prob],
        [prm_min_row["Accuracy (%)"]],
        marker="*",
        s=150,
        color="#d62728",
        zorder=6,
        label=f"PRM Best-of-5 (min): {prm_min_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_total_tok_per_prob],
        [prm_prod_row["Accuracy (%)"]],
        marker="^",
        s=90,
        color="#2ca02c",
        zorder=5,
        label=f"PRM Best-of-5 (product): {prm_prod_row['Accuracy (%)']:.1f}%",
    )
    ax.scatter(
        [prm_total_tok_per_prob],
        [prm_pen_row["Accuracy (%)"]],
        marker="s",
        s=80,
        color="#9467bd",
        zorder=5,
        label=f"PRM Best-of-5 (step_penalty): {prm_pen_row['Accuracy (%)']:.1f}%",
    )
    ax.set_xlabel("Average Output Tokens / Problem (Gen + Judge Tokens)", fontsize=10)
    ax.set_ylabel("Exact-Match Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy vs. Total Output Tokens / Problem", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scaling_tokens.pdf"))
    fig.savefig(os.path.join(fig_dir, "scaling_tokens.png"), dpi=200)
    plt.close(fig)


def run_detailed_analysis(output_dir: str = "outputs", fig_dir: str = "figures") -> dict[str, Any]:
    per_problem_path = os.path.join(output_dir, "per_problem.jsonl")
    records = load_per_problem(per_problem_path)
    with open(os.path.join(output_dir, "results_summary.json"), "r", encoding="utf-8") as f:
        results_summary = json.load(f)
    with open(os.path.join(output_dir, "aggregation_ablation.json"), "r", encoding="utf-8") as f:
        ablation = json.load(f)
    with open(os.path.join(output_dir, "sc_scaling_curve.json"), "r", encoding="utf-8") as f:
        scaling_curve = json.load(f)

    # 1. Bootstrap CIs across methods
    method_keys = [
        ("Greedy (N=1)", "greedy"),
        ("Self-Consistency (N=5)", "sc_5"),
        ("Compute-Matched SC (N=15)", "sc_15"),
        ("PRM Best-of-5 (min)", "prm_min"),
        ("PRM Best-of-5 (product)", "prm_product"),
        ("PRM Best-of-5 (step_penalty, lambda=2.0)", "prm_step_penalty_2.0"),
        ("PRM Best-of-5 (product_logprob)", "prm_product_logprob"),
    ]
    bootstrap_cis = {}
    for label, pkey in method_keys:
        flags = [
            1 if r["predictions"].get(pkey) == r["ground_truth_answer"] else 0
            for r in records
        ]
        mean_acc, ci_low, ci_high = compute_bootstrap_ci(flags, n_resamples=1000, seed=5326)
        bootstrap_cis[label] = {
            "mean": mean_acc,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }

    # 2. Paired win/loss counts
    paired_prm_vs_sc5 = compute_paired_win_loss(records, "prm_min", "sc_5")
    paired_prm_vs_sc15 = compute_paired_win_loss(records, "prm_min", "sc_15")

    # 3. Delimiter compliance rates
    n_probs = len(records)
    greedy_compliant = sum(1 for r in records if r["greedy"]["answer"] is not None)
    all_sc15_cands = [c for r in records for c in r["candidates"]]
    sc_cand_compliant = sum(1 for c in all_sc15_cands if c.get("answer") is not None)

    # 4. 10x Greedy Compute Token Budget
    avg_greedy_tokens = sum(r["bookkeeping"]["greedy_tokens"] for r in records) / n_probs
    budget_10x_greedy = round(10.0 * avg_greedy_tokens, 2)

    # 5. Mine qualitative failure cases and step pool for the 30-step Judge Audit
    judge_fp_candidates = []
    judge_fn_candidates = []
    mode_collapse_cases = []
    invalid_judged_steps = []
    valid_judged_steps = []

    for r in records:
        prob_id = r["id"]
        q = r["question"]
        gt = r["ground_truth_answer"]
        shared_5 = r.get("shared_candidates_5", r["candidates"][:5])

        all_5_wrong = all(c.get("answer") != gt for c in shared_5)
        if all_5_wrong:
            mode_collapse_cases.append({
                "id": prob_id,
                "question": q,
                "ground_truth_answer": gt,
                "candidate_answers": [c.get("answer") for c in shared_5],
                "candidate_0_steps": shared_5[0].get("steps", []) if shared_5 else [],
                "candidate_0_scores": shared_5[0].get("step_scores", []) if shared_5 else [],
            })

        for c_idx, cand in enumerate(shared_5):
            c_ans = cand.get("answer")
            c_steps = cand.get("steps", [])
            c_scores = cand.get("step_scores", [])
            c_rats = cand.get("rationales", [])

            # Candidate reached wrong answer, yet judge marked every single step VALID (+1.0)
            if c_ans is not None and c_ans != gt and c_scores and all(s == 1.0 for s in c_scores):
                judge_fp_candidates.append({
                    "id": prob_id,
                    "candidate_index": c_idx,
                    "question": q,
                    "ground_truth_answer": gt,
                    "candidate_answer": c_ans,
                    "steps": c_steps,
                    "step_scores": c_scores,
                    "rationales": c_rats,
                })

            # Candidate reached the right answer, yet judge marked at least one step INVALID (-1.0)
            if c_ans == gt and c_scores and any(s == -1.0 for s in c_scores):
                judge_fn_candidates.append({
                    "id": prob_id,
                    "candidate_index": c_idx,
                    "question": q,
                    "ground_truth_answer": gt,
                    "candidate_answer": c_ans,
                    "steps": c_steps,
                    "step_scores": c_scores,
                    "rationales": c_rats,
                })

            for s_idx, (st, sc) in enumerate(zip(c_steps, c_scores)):
                rat = c_rats[s_idx] if s_idx < len(c_rats) else ""
                step_entry = {
                    "id": prob_id,
                    "candidate_index": c_idx,
                    "step_number": s_idx + 1,
                    "question": q,
                    "ground_truth_answer": gt,
                    "candidate_answer": c_ans,
                    "candidate_correct": (c_ans == gt),
                    "preceding_steps": c_steps[:s_idx],
                    "step_text": st,
                    "judge_score": sc,
                    "judge_verdict": "VALID" if sc > 0 else "INVALID",
                    "judge_rationale": rat,
                }
                if sc < 0:
                    invalid_judged_steps.append(step_entry)
                else:
                    valid_judged_steps.append(step_entry)

    audit_dump = {
        "total_judged_steps": len(invalid_judged_steps) + len(valid_judged_steps),
        "total_invalid_judged_steps": len(invalid_judged_steps),
        "total_valid_judged_steps": len(valid_judged_steps),
        "judge_fp_candidates_count": len(judge_fp_candidates),
        "judge_fn_candidates_count": len(judge_fn_candidates),
        "mode_collapse_problems_count": len(mode_collapse_cases),
        "judge_fp_examples": judge_fp_candidates[:15],
        "judge_fn_examples": judge_fn_candidates[:15],
        "mode_collapse_examples": mode_collapse_cases[:15],
        "invalid_steps_sample": invalid_judged_steps[:30],
        "valid_steps_sample": valid_judged_steps[:30],
    }
    with open(os.path.join(output_dir, "judge_audit_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(audit_dump, f, indent=2)

    generate_plots(results_summary, scaling_curve, ablation, fig_dir)

    report_stats = {
        "num_problems": n_probs,
        "bootstrap_cis": bootstrap_cis,
        "paired_prm_min_vs_sc5": paired_prm_vs_sc5,
        "paired_prm_min_vs_sc15": paired_prm_vs_sc15,
        "greedy_delimiter_compliance_pct": round(100.0 * greedy_compliant / n_probs, 2),
        "sampled_candidate_compliance_pct": round(
            100.0 * sc_cand_compliant / max(1, len(all_sc15_cands)), 2
        ),
        "avg_greedy_tokens_per_problem": round(avg_greedy_tokens, 2),
        "budget_10x_greedy_tokens_per_problem": budget_10x_greedy,
    }
    with open(os.path.join(output_dir, "report_analysis_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report_stats, f, indent=2)
    return report_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--fig-dir", type=str, default="figures")
    args = parser.parse_args()
    stats = run_detailed_analysis(args.output_dir, args.fig_dir)
    print(json.dumps(stats, indent=2))
