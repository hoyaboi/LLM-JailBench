"""
HarmBench classifier client. Uses original cais/HarmBench-Llama-2-13b-cls via vLLM.
No API - runs locally on GPU.
"""
from typing import Optional
import sys
from pathlib import Path

# Optional: suppress transformers warnings
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

# Import from original HarmBench (no extra deps from eval_utils for just LLM/tokenizer)
_CLS_PATH = "cais/HarmBench-Llama-2-13b-cls"
_NUM_TOKENS = 512  # same as original evaluate_completions default

_classifier_instance: Optional["HarmBenchClassifier"] = None


class HarmBenchClassifier:
    """
    Original HarmBench classifier: cais/HarmBench-Llama-2-13b-cls via vLLM.
    Matches evaluate_completions.py behavior exactly.
    """

    def __init__(
        self,
        cls_path: str = _CLS_PATH,
        tensor_parallel_size: int = 1,
        num_tokens: int = _NUM_TOKENS,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.cls_path = cls_path
        self.num_tokens = num_tokens

        # Tokenizer for truncating generations (right truncation)
        self.tokenizer = AutoTokenizer.from_pretrained(cls_path)
        self.tokenizer.truncation_side = "right"

        # vLLM model (left truncation for long prompts)
        self.llm = LLM(model=cls_path, tensor_parallel_size=tensor_parallel_size)
        self.llm.llm_engine.tokenizer.tokenizer.truncation_side = "left"

        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    def truncate_generation(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate generation to max_tokens (default 512), same as original evaluate_completions."""
        n = max_tokens if max_tokens is not None else self.num_tokens
        tokenized = self.tokenizer.encode(text, max_length=n, truncation=True)
        return self.tokenizer.decode(tokenized, skip_special_tokens=True)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1,
    ) -> str:
        """
        Run classifier. Builds full [INST] prompt and returns generated text (yes/no).
        Interface compatible with LLMClient for drop-in replacement.
        """
        full_prompt = (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
        outputs = self.llm.generate(
            [full_prompt],
            self.sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text or ""


def get_judge_client() -> HarmBenchClassifier:
    """Return singleton HarmBench classifier (original cais/HarmBench-Llama-2-13b-cls)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HarmBenchClassifier()
    return _classifier_instance
