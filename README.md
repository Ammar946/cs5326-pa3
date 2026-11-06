# CS 5326: Advanced Generative AI and Agents
## Programming Assignment 3: Search over Reasoning and Step-Level Process Reward Models

**Design and Lead:** Muhammad Ahmad Sarfraz

---

### Overview
This assignment explores **test-time compute** and **reasoning search** without modifying model parameters. You will implement:
1. **Step-Structured Generation & Baseline Decoders:** Prompting schema enforcing numbered reasoning steps, robust regex answer parser, greedy decoding ($N=1$), and Self-Consistency majority voting ($N=5$).
2. **Process Reward Model (PRM) Step-Level Verifier:** Step-by-step verification judge that evaluates whether intermediate reasoning steps are `VALID` or `INVALID`.
3. **Path Aggregation Strategies:** Designing rules ($\min$-pooling, product of validity, and step error penalty) to score complete reasoning trajectories.
4. **PRM-Guided Best-of-$N$ Search ($N=5$):** Ranking sampled paths using process rewards and selecting the most sound candidate.
5. **Compute-Matched Self-Consistency ($N_{\text{SC}} \approx 15$):** Comparing whether spending test-time compute on step verification beats pure brute-force sampling at equal compute budget.
6. **Benchmark Evaluation:** Running all configurations across the fixed 50-problem GSM8K subset (`data/gsm8k_test_50.jsonl`).

For complete mathematical definitions, grading rubrics, and instructions, please refer to the assignment manual (`PA3.pdf`) released on the course LMS.

---

### Repository Structure
```
cs5326-pa3/
|-- data/
|   `-- gsm8k_test_50.jsonl          # Fixed 50-problem evaluation benchmark
|-- src/
|   |-- __init__.py
|   |-- generation.py                # Step prompt, greedy, self-consistency
|   |-- extraction.py                # Regex parser for steps and numerical answers
|   |-- prm_judge.py                 # Step-level verifier prompt & scoring loop
|   |-- search.py                    # PRM-guided Best-of-N and budget tracking
|   `-- utils.py                     # Dual-model loading, generate/judge callables, bookkeeping helpers
|-- tests/
|   |-- __init__.py
|   |-- test_public_adapter_smoke.py # Public smoke test suite for grading interface
|   `-- test_token_counting.py       # Offline token count test with batch padding
|-- outputs/                         # Generated predictions and metric dumps
|   |-- results_summary.json         # Automated metrics table across all 6 configurations
|   |-- aggregation_ablation.json    # Aggregation strategies and lambda sweep ablation
|   |-- sc_scaling_curve.json        # Self-consistency scaling curve (N=1..15)
|   `-- per_problem.jsonl            # Detailed per-problem records and step scores
|-- run_pipeline.ipynb               # End-to-end evaluation pipeline notebook (you create this)
|-- submission_adapter.py            # Grading adapter contract (Required)
|-- requirements.txt
|-- pyproject.toml
|-- make_submission.sh               # Submission bundler script
|-- REPORT_TEMPLATE.md               # Scaffold for written report
`-- README.md
```

---

### Getting Started

#### 1. Setup Environment
Kaggle's environment includes preinstalled PyTorch, Transformers, Accelerate, Datasets, and NumPy. On Kaggle, upgrade `bitsandbytes` to a modern release:
```bash
pip install -q -U bitsandbytes
```
*(Note: Pinned legacy installs like `bitsandbytes==0.44.1` break on Kaggle's current image because `bitsandbytes 0.44` imports `triton.ops`, which no longer exists. Upgrading to unpinned `bitsandbytes>=0.45.5` resolves this).*

#### 2. Run Public Smoke Tests
The repository includes a public test suite to verify your adapter signatures and parsing math offline:
```bash
pytest -v tests/test_public_adapter_smoke.py
```
*(Initially, tests will fail with `NotImplementedError` until you implement the required adapter methods).*

#### 3. Execution on Kaggle (GPU T4x2)
Create a new Kaggle notebook, set the **Accelerator** in the top sidebar to **GPU T4x2**, and turn **Internet** to **On**. Clone the repository:
```bash
!git clone https://github.com/ahmad-sarfraz345/cs5326-pa3-starter.git
%cd cs5326-pa3-starter
!pip install -q -U bitsandbytes
```
Load the dual models (1.5B generator in fp16 on cuda:0, 7B judge in 4-bit NF4 on cuda:1):
```python
from src.utils import load_models, make_hf_generate_fn, make_hf_judge_fn

gen_model, gen_tokenizer, judge_model, judge_tokenizer = load_models(
    generator_device="cuda:0",
    judge_device="cuda:1"
)
gen_fn = make_hf_generate_fn(gen_model, gen_tokenizer)
judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)
```
*(Warning: Never use `device_map="auto"` across two GPUs for a single model!)*

#### 4. Packaging Your Submission
When your implementation passes all tests and your benchmark run produces all required artifacts in `outputs/`:
```bash
bash make_submission.sh <your_roll_number>
```
This generates `<your_roll_number>_pa3.zip` containing `src/`, `submission_adapter.py`, `run_pipeline.ipynb` (or `run_all.py`), `outputs/per_problem.jsonl`, `outputs/results_summary.json`, `outputs/aggregation_ablation.json`, `outputs/sc_scaling_curve.json`, and `REPORT.pdf`. Upload this archive to LMS.

