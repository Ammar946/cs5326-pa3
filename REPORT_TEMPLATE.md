# CS 5326: Programming Assignment 3 Technical Report
**Student Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Date:** [Date]  

***

## Section 1: Benchmark Results and Compute Scaling (7 Points)

### 1.1 Summary Results Table
Populate with empirical metrics generated from your benchmark evaluation over `data/gsm8k_test_50.jsonl`:

| Method | Accuracy (%) | Generation Calls | Judge Calls | Total LLM Calls | Avg Gen Tokens / Problem | Avg Judge Tokens / Problem | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Greedy (N=1) | | 50 | 0 | 50 | | 0.0 | |
| Self-Consistency (N=5) | | 250 | 0 | 250 | | 0.0 | |
| Compute-Matched SC (N=15) | | 750 | 0 | 750 | | 0.0 | |
| PRM Best-of-5 (min) | | 250 | 250 x K_bar | 250 + 250*K_bar | | | |
| PRM Best-of-5 (product) | | 250 | 250 x K_bar | 250 + 250*K_bar | | | |
| PRM Best-of-5 (step_penalty, lambda=2.0) | | 250 | 250 x K_bar | 250 + 250*K_bar | | | |

*Note: PRM Best-of-5 configurations share identical candidate paths and step scores. In results_summary.json, all three PRM rows report the same numeric Generation Calls, Judge Calls, Total LLM Calls, and token columns. Time attribution rules: Greedy = its own generation time; SC@5 = time to sample the 5 shared candidates; each PRM row (min, product, step_penalty) = SC@5 sampling time + total judging time + its own aggregation/selection time (each row reports the full cost of running that method, even though judging is computed once and cached); SC@15 = SC@5 sampling time + time to sample the 10 additional candidates.*

### 1.2 Scaling Plots
* Include two clear scaling plots:
  1. **Accuracy vs. Total LLM Calls**: Plot Self-Consistency accuracy across candidate budgets $N = 1 \dots 15$ from `outputs/sc_scaling_curve.json`, and overlay distinct points marking PRM Best-of-5 methods.
  2. **Accuracy vs. Total Output Tokens**: Plot accuracy against total generated + judge tokens per problem to reflect actual compute costs.

### 1.3 Statistical Analysis and Discussion
* **Paired Win / Loss Counts**:
  - PRM-BoN (min) vs. Self-Consistency (N=5): Problems where PRM won: [X], SC won: [Y], Both agreed: [Z].
  - PRM-BoN (min) vs. Compute-Matched SC (N=15): Problems where PRM won: [X], SC won: [Y], Both agreed: [Z].
* **95% Bootstrap Confidence Intervals**: Compute empirical 95% bootstrap confidence intervals for accuracy across methods (1,000 resamples over problems, seed 5326).
* **Discussion**: Does PRM-guided Best-of-5 outperform candidate-matched SC (N=5)? Does it outperform compute-matched SC (N=15)? Do observed differences exceed statistical noise?

***

## Section 2: Path Score Aggregation Ablation (6 Points)

### 2.1 Aggregation Ablation Table
Serialize your empirical findings to `outputs/aggregation_ablation.json` and report them below:

| Strategy | Lambda | Accuracy (%) | Tie Rate (%) |
| :--- | :---: | :---: | :---: |
| min | null | | |
| product | null | | |
| step_penalty | 0.0 | | |
| step_penalty | 0.5 | | |
| step_penalty | 2.0 | | |

*Tie Rate is defined as the percentage of benchmark problems where more than one candidate tied for the maximum path score (num_tied > 1).*

### 2.2 Analytical Breakdown
* Explain why the product strategy behaves similarly to min-pooling when using discrete binary scores.
* Analyze the filter-then-vote interaction: how does the majority-vote tie-breaking rule interact with min-pooling when multiple candidates achieve a score of +1.0?
* *(Optional Bonus, +3 Points)*: If you implemented the `product_logprob` variant in `src/`, report its performance and compare calibrated token probabilities against discrete categorical judgments.

***

## Section 3: Failure Mode and Judge Error Analysis (7 Points)

### 3.1 Qualitative Failure Cases
Identify three concrete failure examples from your GSM8K benchmark evaluation. If a failure category does not occur in your run, present the closest example and explain its absence:

#### Case 1: Judge False Positive (Flawed step marked VALID)
* **Problem ID / Question:**
* **Flawed Step:**
* **Judge Verdict & Rationale:**
* **Root Cause Analysis:** Why did the 7B judge fail to detect the mathematical or logical error?

#### Case 2: Judge False Negative (Valid step marked INVALID)
* **Problem ID / Question:**
* **Valid Step Flagged as Invalid:**
* **Judge Verdict & Rationale:**
* **Root Cause Analysis:** Why did the judge incorrectly penalize a mathematically sound step?

#### Case 3: Mode Collapse / Shared Flawed Premise
* **Problem ID / Question:**
* **Common Flaw Shared Across Candidates:**
* **Root Cause Analysis:** Why was search unable to recover the correct solution?

### 3.2 Human Audit of Judge Decisions
Hand-label at least **30 individual judged steps** from your benchmark runs (including at least **10 steps marked INVALID** by the judge):

| Metric | Count / Value |
| :--- | :---: |
| True Positives (Judge: INVALID, Human: INVALID) | |
| False Positives (Judge: INVALID, Human: VALID) | |
| True Negatives (Judge: VALID, Human: VALID) | |
| False Negatives (Judge: VALID, Human: INVALID) | |
| **Judge Precision (for INVALID detection)** | **%** |
| **Judge Recall (for INVALID detection)** | **%** |

*Note: If any cell in your confusion matrix is zero or a particular error category is absent from your sample, provide an explanation based on your model's error distribution.*

### 3.3 Discussion Question
How does utilizing a substantially stronger judge model (7B) than generator policy (1.5B) influence the types of reasoning errors caught, and what failure modes would you anticipate if the 1.5B generator were tasked with self-verification?

***

## Section 4: Test-Time Compute Efficiency Analysis (3 Points)
* **Defining 10x Greedy Compute**: Calculate your exact token budget defined as $10\times$ greedy generated tokens. On the spending side, count both generated tokens and judge tokens.
* **Strategic Recommendation**: Under this fixed compute budget, would you recommend allocating inference compute to pure repeated sampling (Self-Consistency with large $N$) or PRM-guided search? Justify your decision using your empirical scaling curve and measured token expenditures.

***

## Section 5: Limitations and Reproducibility (2 Points)
* **Hardware & Runtime Environment**: GPU model (e.g. Kaggle GPU T4x2), VRAM utilization, total wall-clock runtime, library versions, and execution of the per-problem seed scheme (`5326 + problem_index`).
* **Methodological Limitations**: Contrast this zero-shot prompted PRM stand-in with true process reward models trained directly on dense human step-level verifications (e.g. PRM800K).
