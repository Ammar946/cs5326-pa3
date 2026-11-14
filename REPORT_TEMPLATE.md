# CS 5326: Programming Assignment 3 Technical Report
**Student Name:** Muhammad Ammar Shahbaz  
**GitHub Repository:** `Ammar946/cs5326-pa3`  
**Date:** Fall 2026  

***

## Section 1: Benchmark Results and Compute Scaling (7 Points)

### 1.1 Summary Results Table
Empirical metrics generated from our benchmark evaluation over `data/gsm8k_test_50.jsonl` (`outputs/results_summary.json` and `outputs/bonus_product_logprob.json`, $\bar{K} = 4.168$ steps/candidate across 250 shared candidates):

| Method | Accuracy (%) | 95% Boot. CI (%) | Generation Calls | Judge Calls | Total LLM Calls | Avg Gen Tokens / Problem | Avg Judge Tokens / Problem | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Greedy (N=1) | 66.00 | [52.00, 78.00] | 50 | 0 | 50 | 218.80 | 0.00 | 477.08 |
| Self-Consistency (N=5) | 72.00 | [60.00, 84.00] | 250 | 0 | 250 | 922.32 | 0.00 | 556.61 |
| Compute-Matched SC (N=15) | 76.00 | [62.00, 86.00] | 750 | 0 | 750 | 2828.26 | 0.00 | 1250.33 |
| **PRM Best-of-5 (min)** | **80.00** | **[68.00, 90.00]** | 250 | 1042 | 1292 | 922.32 | 620.06 | 3037.35 |
| PRM Best-of-5 (product) | 72.00 | [58.00, 84.00] | 250 | 1042 | 1292 | 922.32 | 620.06 | 3037.35 |
| **PRM Best-of-5 (step_penalty, lambda=2.0)** | **80.00** | **[68.00, 90.00]** | 250 | 1042 | 1292 | 922.32 | 620.06 | 3037.35 |
| PRM Best-of-5 (product_logprob) *[Bonus]* | 78.00 | [66.00, 88.00] | 250 | 1042 | 1292 | 922.32 | 620.06 | 3037.35 |

### 1.2 Scaling Plots
* Standalone PDF scaling plots are generated in `figures/scaling_calls.pdf` and `figures/scaling_tokens.pdf` (and embedded as vector `pgfplots` in `REPORT.pdf`, Figure 1):
  1. **Accuracy vs. Total LLM Calls** (`figures/scaling_calls.pdf`): Self-Consistency accuracy across $N = 1 \dots 15$ (`outputs/sc_scaling_curve.json`) overlaid with Greedy ($N=1$) and PRM Best-of-5 variants at $1,292$ calls.
  2. **Accuracy vs. Total Output Tokens** (`figures/scaling_tokens.pdf`): Accuracy vs. average output tokens per problem ($\text{generated\_tokens} + \text{judge\_tokens}$), showing PRM Best-of-5 (`min`/`step_penalty`) reaching $80.0\%$ at $1,542.38$ tok/problem vs. $2,828.26$ tok/problem ($76.0\%$) for SC@15.

### 1.3 Statistical Analysis and Discussion
* **Paired Win / Loss Counts**:
  - **PRM-BoN (min) vs. Self-Consistency (N=5)**: Problems where PRM won: **4** (`gsm8k_test_03`, `05`, `45`, `46`), SC won: **0**, Both agreed: **46** (36 both correct, 10 both wrong).
  - **PRM-BoN (min) vs. Compute-Matched SC (N=15)**: Problems where PRM won: **3** (`gsm8k_test_03`, `05`, `45`), SC won: **1** (`gsm8k_test_21`), Both agreed: **46** (37 both correct, 9 both wrong).
* **95% Bootstrap Confidence Intervals**: Computed over 1,000 resamples with seed 5326 (see table above and `outputs/report_analysis_stats.json`).
* **Discussion**: PRM-guided Best-of-5 (`min` and `step_penalty`) outperforms candidate-matched SC ($N=5$) by **+8.00%** ($80.0\%$ vs. $72.0\%$) and outperforms compute-matched SC ($N=15$) by **+4.00%** ($80.0\%$ vs. $76.0\%$). Because $M=50$ yields a binomial standard error of $\approx 6.0\%$, unpaired 95% bootstrap CIs overlap across methods; however, paired comparison over the exact same 5 candidate paths shows a 4-to-0 win/loss advantage over SC@5 ($p = 0.0625$ one-sided sign test), proving that whenever PRM (`min`) overrides the SC@5 majority vote, PRM is right 100% of the time ($4/4$).

***

## Section 2: Path Score Aggregation Ablation (6 Points)

### 2.1 Aggregation Ablation Table
Empirical findings serialized to `outputs/aggregation_ablation.json` (plus `outputs/bonus_product_logprob.json`):

| Strategy | Lambda | Accuracy (%) | Tie Rate (%) |
| :--- | :---: | :---: | :---: |
| min | null | **80.00** | 84.00 (42/50) |
| product | null | 72.00 | 52.00 (26/50) |
| step_penalty | 0.0 | **80.00** | 78.00 (39/50) |
| step_penalty | 0.5 | **80.00** | 78.00 (39/50) |
| step_penalty | 2.0 | **80.00** | 78.00 (39/50) |
| product_logprob *[Bonus]* | null | 78.00 | 30.00 (15/50) |

### 2.2 Analytical Breakdown
* **Discrete `product` vs. `min`-pooling**: Under binary step scores $r_t \in \{-1.0, +1.0\}$, clamping $(r_t+1)/2$ to $[0.01, 0.99]$ yields $S_{\text{prod}}(\tau) = 0.99^T$ for any all-valid path ($k=0$ invalid steps). Because $0.99^2 - 0.99^3 = 0.009801 \gg 10^{-9}$, two all-valid paths tie under `product` if and only if they have the exact same step count $T$. This reduces the tie rate from **84.0%** (`min`) to **52.0%** (`product`) and systematically favors compressed 2–3 step paths that skip intermediate reasoning over thorough 4–6 step derivations (causing 4 errors on `gsm8k_test_02`, `17`, `30`, `48`).
* **Invariance of `step_penalty` across $\lambda \in \{0.0, 0.5, 2.0\}$**: Because $S_{\text{penalty}}(T, k) = 1 - k(\lambda + 2/T)$, every all-valid path ($k=0$) scores $1.0$ regardless of $T$ or $\lambda$, matching `min` on all 46 problems that have at least one valid candidate. Only on the 4 problems where all 5 candidates had $k \ge 1$ invalid steps does `step_penalty` differentiate candidates by $(T, k)$ (reducing tie rate from 84.0% to 78.0%).
* **Filter-then-Vote Interaction**: Under `min`-pooling, all $k=0$ candidates tie at $+1.0$ (84.0% of problems, averaging 3.62 tied candidates/problem), and `select_best_path` runs majority voting over the surviving valid subset. Without majority-vote tie-breaking (lowest index wins), accuracy drops from **80.0%** to **74.0%** because isolated Judge False Positives at index 0 are outvoted by valid peers under filter-then-vote.
* **Optional Bonus (`product_logprob`, +3 Points)**: Using first-token logit differences $p_t = \sigma(z_{\text{VALID},t} - z_{\text{INVALID},t}) \in [10^{-4}, 0.9999]$ from `model.generate(..., output_scores=True)` achieves **78.00% accuracy** (**+6.00% over discrete `product`**) with a 30.0% tie rate. Confident valid steps have $p_t \approx 0.9999$ (eliminating the artificial length penalty $0.99^T$), while hand-wavy compressed steps have lower confidence ($p_t \approx 0.80$), rescuing `gsm8k_test_02`, `17`, and `48`.

***

## Section 3: Failure Mode and Judge Error Analysis (7 Points)

### 3.1 Qualitative Failure Cases

#### Case 1: Judge False Positive (Flawed step marked VALID)
* **Problem ID / Question:** `gsm8k_test_06` (GT: `64`, Selected: `38`) — *"Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"*
* **Flawed Step (Cand 1, Step 3):** `"Kylar wants to buy a total of 16 glasses, and he is buying every second glass at a 60% discount. Therefore, there will be (16 - 1) / 2 = 7 glasses that cost $5 each and 1 glass that costs $3 each."`
* **Judge Verdict & Rationale:** `Verdict: VALID` ($r_3 = +1.0, p_3 = 0.9998$). `"Rationale: The current step correctly calculates the number of discounted and full-priced glasses based on the given conditions."`
* **Root Cause Analysis:** Step 3 evaluates $(16-1)/2 = 7.5$ as $7$ and partitions 16 glasses into $7 + 1 = 8$ glasses total instead of 8 full-price and 8 discounted glasses. The 4-bit 7B judge is misled by the authoritative formula structure without checking the conservation sum $7 + 1 \ne 16$.

#### Case 2: Judge False Negative (Valid step marked INVALID)
* **Problem ID / Question:** `gsm8k_test_26` (GT: `2`, Cand 3 Answer: `2`) — *"Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?"*
* **Valid Step Flagged as Invalid (Cand 3, Step 6):** `"Calculate the number of boxes of pizza Marie ordered. Number of boxes of pizza = Total cost of pizza / Cost per box of pizza = $17 / $8.50 = 2 boxes."`
* **Judge Verdict & Rationale:** `Verdict: INVALID` ($r_6 = -1.0, p_6 = 0.0029$). `"Rationale: The total cost of pizza calculated in Step 5 should be $17, but the division $17 / $8.50 does not yield an integer number of boxes, indicating an error in the problem setup or a need for rounding."`
* **Root Cause Analysis:** Because 17 is an odd prime and 8.50 is a decimal, the 4-bit quantized 7B judge heuristically assumes $17 / 8.50$ is non-integral without verifying that $8.50 \times 2 = 17.0$.

#### Case 3: Mode Collapse / Shared Flawed Premise
* **Problem ID / Question:** `gsm8k_test_13` (GT: `13`, Candidate Answers: `['12', '12', '11', '12', '12']`) — *"Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?"*
* **Common Flaw Shared Across Candidates:** Four of five candidates compute net annual profit $10.50 - 3.00 = 7.50$, divide $90 / 7.50 = 12$, and output `12` (exact break-even where cumulative profit is $\$0$) instead of year `13` (when cumulative profit becomes strictly positive).
* **Root Cause Analysis:** Both the 1.5B generator and 7B judge conflate "starts earning money" with break-even time ($\text{Pass@5} = 0\%$), showing that Best-of-$N$ search cannot recover solutions outside the support of the generator policy.

### 3.2 Human Audit of Judge Decisions
Stratified 30-step human audit (`outputs/human_judge_audit_30.json`: 15 steps marked `INVALID` + 15 steps marked `VALID`):

| Metric | Count / Value |
| :--- | :---: |
| True Positives (Judge: INVALID, Human: INVALID) | 12 |
| False Positives (Judge: INVALID, Human: VALID) | 3 |
| True Negatives (Judge: VALID, Human: VALID) | 13 |
| False Negatives (Judge: VALID, Human: INVALID) | 2 |
| **Judge Precision (for INVALID detection)** | **80.00% (12/15)** |
| **Judge Recall (for INVALID detection)** | **85.71% (12/14)** |

### 3.3 Discussion Question
Utilizing a $4.7\times$ larger verifier (`Qwen2.5-7B-Instruct`) than the generator (`Qwen2.5-1.5B-Instruct`) breaks self-verification symmetry, allowing the judge to catch multi-clause relational misreadings (85.71% recall). If the 1.5B generator performed self-verification, shared parametric blind spots and sycophantic confirmation bias would cause it to rubber-stamp its own high-probability misconceptions as `VALID`.

***

## Section 4: Test-Time Compute Efficiency Analysis (3 Points)
* **Defining 10x Greedy Compute**: Greedy ($N=1$) averages **218.80 output tokens/problem**, so the $10\times$ Greedy budget is **$2,188.00$ output tokens/problem** ($109,400$ total tokens across 50 problems).
* **PRM-5 vs. SC Budget Utilization**: PRM Best-of-5 consumes $922.32$ (gen) $+ 620.06$ (judge) $= \mathbf{1,542.38}$ **output tokens/problem** ($7.05\times$ Greedy), fitting well inside the $2,188.00$ budget, whereas SC@15 consumes $2,828.26$ tokens/problem ($12.93\times$ Greedy, exceeding budget). Under $2,188.00$ tokens/problem, pure Self-Consistency can afford up to $N=11$ ($2,059.08$ tokens/problem).
* **Strategic Recommendation**: Under the $10\times$ Greedy output token budget, we recommend **PRM-guided Best-of-5 (`min` or `step_penalty`)**: it achieves **80.00%** accuracy at $1,542.38$ tokens/problem, beating token-matched SC@$N=8$ ($1,500.12$ tokens, $76.00\%$) by $+4.00\%$ and using $25.1\%$ fewer output tokens than SC@$N=11$ ($2,059.08$ tokens, $80.00\%$) while avoiding high-$N$ majority-vote regression ($76.00\%$ at $N=15$).

***

## Section 5: Limitations and Reproducibility (2 Points)
* **Hardware & Runtime Environment**: Kaggle dual NVIDIA Tesla T4 GPUs ($2 \times 16\text{ GB}$ VRAM); `Qwen2.5-1.5B-Instruct` in `float16` on `cuda:0` ($\approx 3.1\text{ GB}$) and `Qwen2.5-7B-Instruct` in 4-bit `NF4` on `cuda:1` ($\approx 5.5\text{ GB}$). Per-problem deterministic seeding `5326 + problem_index`. Total wall-clock runtime: **$4,208.13\text{s}$ ($70.14\text{ min}$)**.
* **Methodological Limitations**: Unlike discriminative PRMs trained on dense human step labels (e.g., PRM800K) or lookahead value functions $V^\pi(x, s_{1:t})$, a zero-shot prompted LLM verifier (1) assesses only myopic local validity rather than future solvability, and (2) requires $\sim 30$ autoregressive rationale tokens per step instead of a single classification head forward pass.
