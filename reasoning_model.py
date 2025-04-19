# see https://github.com/simplescaling/s1

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_TOKENS_THINKING = 32000

class ReasoningModel:
    def __init__(self, model_name: str = "simplescaling/s1-32B", tensor_parallel_size: int = 1):
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization = 0.90)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # stop generating text after the end token or after a step of reasoning
        self.stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        self.stop_strings = ["\n\n", "<|im_start|>"]
        
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=self.stop_token_ids,
            stop=self.stop_strings,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
        )

        self.total_tokens = 0

    def set_stop_conditions(self, stop_tokens: list[str], stop_strings: list[str]) -> None:
        """
        Set custom stop tokens/strings for the model.
        
        Args:
            stop_tokens: List of strings that will be used as stop tokens
            stop_strings: List of strings that will be used as stop strings
        """
        stop_ids = []
        for token in stop_tokens:
            stop_ids.extend(self.tokenizer(token)["input_ids"])
        self.stop_token_ids = stop_ids

        self.stop_strings = stop_strings
        
        # Update sampling parameters with new stop tokens
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=self.stop_token_ids,
            stop=self.stop_strings,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
        )
    
    def get_total_tokens(self) -> int:
        return self.total_tokens

    def generate_response(self, prompt: str, min_tokens=0) -> str:
        ignore_str = 'Wait'
        full_output = ''
        num_output_tokens = 0
        while True:
            output = self.model.generate(prompt, sampling_params=self.sampling_params, use_tqdm=False)
            full_output += output[0].outputs[0].text
            num_output_tokens = len(self.tokenizer(full_output)["input_ids"])

            if num_output_tokens >= min_tokens:
                break

            prompt += output[0].outputs[0].text + ignore_str

        self.total_tokens += num_output_tokens

        print('\t\t' + repr(full_output))    
        return output[0].outputs[0].text