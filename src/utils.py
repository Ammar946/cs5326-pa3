"""
src/utils.py - Model loading, generation constants, seed management, and factories.
"""
import json
import math
import os
from typing import Any, Callable

# Global random seed specified by course policy
GLOBAL_SEED = 5326

# Sampling & decoding hyperparameter configurations
GREEDY_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

JUDGE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 96,
}


def set_seed_for_problem(problem_index: int) -> None:
    """
    Sets a deterministic seed for problem_index (5326 + problem_index).
    Guarantees reproducibility and safe resumability after interruptions.
    """
    try:
        import transformers
        transformers.set_seed(GLOBAL_SEED + problem_index)
    except ImportError:
        import random
        random.seed(GLOBAL_SEED + problem_index)


def load_models(generator_device: str = "cuda:0", judge_device: str = "cuda:1"):
    """
    Loads Qwen2.5-1.5B-Instruct in float16 (Generator) and
    Qwen2.5-7B-Instruct in 4-bit NF4 (Judge).
    
    Hardware Mapping:
      - Kaggle GPU T4x2: generator on cuda:0, judge on cuda:1.
      - Single GPU: both models placed on cuda:0.
      - Warning: Never use device_map="auto" across two GPUs for a single model!
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        g_dev = "cuda:0" if num_gpus == 1 else "cpu"
        j_dev = "cuda:0" if num_gpus == 1 else "cpu"
    else:
        g_dev = generator_device
        j_dev = judge_device

    # 1. Generator: Qwen/Qwen2.5-1.5B-Instruct in float16 (~3.1 GB VRAM)
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_id)
    if gen_tokenizer.pad_token_id is None:
        gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

    gen_device_map = {"": g_dev} if g_dev != "cpu" else None
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id,
        torch_dtype=torch.float16 if g_dev != "cpu" else torch.float32,
        device_map=gen_device_map,
    )
    if g_dev == "cpu":
        gen_model = gen_model.to("cpu")

    # 2. Judge: Qwen/Qwen2.5-7B-Instruct in 4-bit NF4 (~5.5 GB VRAM)
    judge_id = "Qwen/Qwen2.5-7B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    judge_device_map = {"": j_dev} if j_dev != "cpu" else None
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_id,
        quantization_config=bnb_config if j_dev != "cpu" else None,
        torch_dtype=torch.float16 if j_dev != "cpu" else torch.float32,
        device_map=judge_device_map,
    )
    if j_dev == "cpu":
        judge_model = judge_model.to("cpu")

    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def _get_eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Extracts all EOS token IDs from model.generation_config and tokenizer."""
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        cfg_eos = getattr(model.generation_config, "eos_token_id", None)
        if cfg_eos is not None:
            if isinstance(cfg_eos, int):
                eos_ids.add(cfg_eos)
            elif isinstance(cfg_eos, (list, tuple, set)):
                eos_ids.update(cfg_eos)
    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            if isinstance(tok_eos, int):
                eos_ids.add(tok_eos)
            elif isinstance(tok_eos, (list, tuple, set)):
                eos_ids.update(tok_eos)
    return eos_ids


def _slice_to_first_eos(gen_tokens: Any, eos_ids: set[int]) -> Any:
    """Slices token sequence up to and including the first EOS token."""
    if hasattr(gen_tokens, "tolist"):
        ids = gen_tokens.tolist()
    else:
        ids = list(gen_tokens)
    first_eos_idx = None
    for idx, tok in enumerate(ids):
        if tok in eos_ids:
            first_eos_idx = idx
            break
    if first_eos_idx is not None:
        return gen_tokens[:first_eos_idx + 1]
    return gen_tokens


def _extract_verdict_logprob(
    gen_tokens_valid: Any,
    step_scores_tuple: Any,
    valid_token_pairs: list[tuple[int, int]],
) -> float | None:
    """
    Computes continuous step validity probability p_t from the judge's raw logits
    at the verdict token:
        p_t = exp(z_VALID) / (exp(z_VALID) + exp(z_INVALID))
            = 1 / (1 + exp(z_INVALID - z_VALID))
    """
    if not step_scores_tuple or not valid_token_pairs:
        return None

    try:
        token_ids = gen_tokens_valid.tolist() if hasattr(gen_tokens_valid, "tolist") else list(gen_tokens_valid)
        target_step_idx = None
        matched_pair = valid_token_pairs[0]

        all_valid_ids = {p[0] for p in valid_token_pairs}
        all_invalid_ids = {p[1] for p in valid_token_pairs}

        for idx, tok_id in enumerate(token_ids):
            if idx >= len(step_scores_tuple):
                break
            if tok_id in all_valid_ids or tok_id in all_invalid_ids:
                target_step_idx = idx
                for v_id, inv_id in valid_token_pairs:
                    if tok_id == v_id or tok_id == inv_id:
                        matched_pair = (v_id, inv_id)
                        break
                break

        if target_step_idx is None:
            return None

        logits_step = step_scores_tuple[target_step_idx][0]
        v_id, inv_id = matched_pair
        z_valid = float(logits_step[v_id].item() if hasattr(logits_step[v_id], "item") else logits_step[v_id])
        z_invalid = float(logits_step[inv_id].item() if hasattr(logits_step[inv_id], "item") else logits_step[inv_id])

        diff = max(-50.0, min(50.0, z_valid - z_invalid))
        prob = 1.0 / (1.0 + math.exp(-diff))
        return max(1e-4, min(0.9999, prob))
    except Exception:
        return None


def make_hf_generate_fn(model: Any, tokenizer: Any) -> Callable[[str, int, dict], list[dict]]:
    """
    Factory creating a GenerateFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw instruction prompts.
    """
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(gen_kwargs)
        if not kwargs.get("do_sample", False):
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None

        if num_samples > 1 and kwargs.get("do_sample", True):
            kwargs["num_return_sequences"] = num_samples

        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(model, tokenizer)
        results = []
        for i in range(output_ids.shape[0]):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
            text = tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(gen_tokens_valid)})
        return results

    return generate_fn


def make_hf_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[str], dict]:
    """
    Factory creating a JudgeFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw judge prompts.
    When used with a full HuggingFace tokenizer/model, also extracts continuous
    step confidence 'valid_prob' from the verdict token's logits (Optional Bonus).
    """
    valid_token_pairs: list[tuple[int, int]] = []
    supports_encode = hasattr(judge_tokenizer, "encode")
    if supports_encode:
        try:
            for v_str, inv_str in [(" VALID", " INVALID"), ("VALID", "INVALID"), (" valid", " invalid")]:
                v_ids = judge_tokenizer.encode(v_str, add_special_tokens=False)
                inv_ids = judge_tokenizer.encode(inv_str, add_special_tokens=False)
                if v_ids and inv_ids:
                    pair = (int(v_ids[0]), int(inv_ids[0]))
                    if pair not in valid_token_pairs:
                        valid_token_pairs.append(pair)
        except Exception:
            valid_token_pairs = []

    def judge_fn(judge_prompt: str) -> dict:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": judge_prompt}]
        formatted = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = judge_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(judge_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)
        if valid_token_pairs:
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True

        with no_grad_ctx():
            raw_output = judge_model.generate(**inputs, **kwargs)

        if hasattr(raw_output, "sequences"):
            output_ids = raw_output.sequences
            step_scores_tuple = getattr(raw_output, "scores", None)
        else:
            output_ids = raw_output
            step_scores_tuple = None

        eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
        gen_tokens = output_ids[0][prompt_len:]
        gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
        text = judge_tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)

        result: dict[str, Any] = {"text": text, "num_tokens": len(gen_tokens_valid)}
        if step_scores_tuple is not None and valid_token_pairs:
            v_prob = _extract_verdict_logprob(gen_tokens_valid, step_scores_tuple, valid_token_pairs)
            if v_prob is not None:
                result["valid_prob"] = float(v_prob)
        return result

    return judge_fn


def save_results_summary(results_list: list[dict[str, Any]], filepath: str = "outputs/results_summary.json") -> None:
    """
    Saves results_summary.json conforming strictly to the assignment schema:
    Keys: 'Method', 'Accuracy (%)', 'Generation Calls', 'Judge Calls', 'Total LLM Calls',
          'Avg Gen Tokens / Problem', 'Avg Judge Tokens / Problem', 'Total Time (s)'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)


def save_aggregation_ablation(ablation_list: list[dict[str, Any]], filepath: str = "outputs/aggregation_ablation.json") -> None:
    """
    Saves aggregation_ablation.json:
    List of {'Strategy': str, 'Lambda': float | None, 'Accuracy (%)': float, 'Tie Rate (%)': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ablation_list, f, indent=2)


def save_sc_scaling_curve(curve_list: list[dict[str, Any]], filepath: str = "outputs/sc_scaling_curve.json") -> None:
    """
    Saves sc_scaling_curve.json:
    List of {'N': int, 'Accuracy (%)': float, 'Avg Gen Tokens / Problem': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(curve_list, f, indent=2)
