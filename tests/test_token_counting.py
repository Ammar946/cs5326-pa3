"""
Unit test for token-counting and padding truncation in make_hf_generate_fn and make_hf_judge_fn.
Runs offline with real PyTorch if available, or with lightweight mock tensors if torch is absent.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MockTensor:
    def __init__(self, data):
        self.data = data
        if isinstance(data, list) and data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        elif isinstance(data, list):
            self.shape = (len(data),)
        else:
            self.shape = ()

    def tolist(self):
        return self.data

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MockTensor(self.data[item])
        res = self.data[item]
        if isinstance(res, list):
            return MockTensor(res)
        return res

    def __len__(self):
        return len(self.data)


def create_tensor(data):
    if HAS_TORCH:
        return torch.tensor(data)
    return MockTensor(data)


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 151645
        self.pad_token_id = 151643

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, text, return_tensors=None):
        # 3-token prompt: [101, 102, 103]
        return {"input_ids": create_tensor([[101, 102, 103]])}

    def decode(self, tokens, skip_special_tokens=True):
        if hasattr(tokens, "tolist"):
            ids = tokens.tolist()
        else:
            ids = list(tokens)
        return " ".join(f"tok_{t}" for t in ids)


class FakeGenConfig:
    def __init__(self):
        # List of EOS token IDs (e.g. multiple EOS candidates)
        self.eos_token_id = [151645, 151643]


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.generation_config = FakeGenConfig()

    def generate(self, input_ids, **kwargs):
        # Prompt len is 3 (tokens 101, 102, 103).
        # Sequence 0: 2 generated tokens (10, 20) + EOS (151645) + 2 PAD tokens (999, 999) -> 3 valid tokens
        # Sequence 1: 4 generated tokens (10, 20, 30, 40) + EOS (151645) -> 5 valid tokens
        seq0 = [101, 102, 103, 10, 20, 151645, 999, 999]
        seq1 = [101, 102, 103, 10, 20, 30, 40, 151645]
        return create_tensor([seq0, seq1])


class FakeJudgeModel:
    def __init__(self):
        self.device = "cpu"
        self.generation_config = FakeGenConfig()

    def generate(self, input_ids, **kwargs):
        # Prompt len is 3. Output has 2 valid tokens (50, 60) + EOS (151645) + 1 PAD (999)
        seq = [101, 102, 103, 50, 60, 151645, 999]
        return create_tensor([seq])


def test_padded_batch_token_counting_generate_fn():
    from src.utils import make_hf_generate_fn
    model = FakeModel()
    tokenizer = FakeTokenizer()
    gen_fn = make_hf_generate_fn(model, tokenizer)
    results = gen_fn("Solve 2+2", num_samples=2, gen_kwargs={"do_sample": True})

    assert len(results) == 2
    # Sequence 0 had 2 tokens + EOS + 2 PAD: should count 3 tokens, not 5
    assert results[0]["num_tokens"] == 3
    assert "tok_999" not in results[0]["text"]
    assert "tok_10" in results[0]["text"] and "tok_20" in results[0]["text"]

    # Sequence 1 had 4 tokens + EOS: should count 5 tokens
    assert results[1]["num_tokens"] == 5
    assert "tok_40" in results[1]["text"]


def test_padded_batch_token_counting_judge_fn():
    from src.utils import make_hf_judge_fn
    judge_model = FakeJudgeModel()
    judge_tokenizer = FakeTokenizer()
    judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)
    res = judge_fn("Judge step")

    # Output had 2 tokens + EOS + 1 PAD: should count 3 tokens, not 4
    assert res["num_tokens"] == 3
    assert "tok_999" not in res["text"]
    assert "tok_50" in res["text"] and "tok_60" in res["text"]


if __name__ == "__main__":
    test_padded_batch_token_counting_generate_fn()
    test_padded_batch_token_counting_judge_fn()
    print("All token-counting tests passed successfully!")
