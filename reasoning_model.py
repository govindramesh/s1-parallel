# see https://github.com/simplescaling/s1

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_TOKENS_THINKING = 32000

class ReasoningModel:
    def __init__(self, model_name: str = "simplescaling/s1-32B", tensor_parallel_size: int = 1):
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization = 0.90)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # stop generating text after the end token or after a step of reasoning
        self.stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"] + self.tokenizer("\n\n")["input_ids"]
        
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=self.stop_token_ids,
        )

    def set_stop_tokens(self, stop_tokens: list[str]) -> None:
        """
        Set custom stop tokens for the model.
        
        Args:
            stop_tokens: List of strings that will be used as stop tokens
        """
        stop_ids = []
        for token in stop_tokens:
            stop_ids.extend(self.tokenizer(token)["input_ids"])
        self.stop_token_ids = stop_ids
        
        # Update sampling parameters with new stop tokens
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=self.stop_token_ids,
        )

    def generate_response(self, prompt: str, min_tokens=0) -> str:
        ignore_str = 'Wait'
        full_output = ''
        num_output_tokens = 0
        while True:
            output = self.model.generate(prompt, sampling_params=self.sampling_params)
            full_output += output[0].outputs[0].text
            num_output_tokens = len(self.tokenizer(full_output)["input_ids"])

            if num_output_tokens >= min_tokens:
                break

            prompt += output[0].outputs[0].text + ignore_str
            
        return output[0].outputs[0].text.split("\n\n")[-1].strip()