# see https://github.com/simplescaling/s1

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_TOKENS_THINKING = 32000

class ReasoningModel:
    def __init__(self, model_name: str = "simplescaling/s1-32B", tensor_parallel_size: int = 1):
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # stop generating text after the end token or after a step of reasoning
        self.stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"] + self.tokenizer("\n\n")["input_ids"]
        
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=self.stop_token_ids,
        )

    def generate_response(self, prompt: str) -> str:
        output = self.model.generate(prompt, sampling_params=self.sampling_params)
        return output[0].outputs[0].text.split("\n\n")[-1].strip()