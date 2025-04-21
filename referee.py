import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class RefereeModel:
    def __init__(self, model_name: str = "google/gemma-2-9b-it", device_num: int = 0):
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.bfloat16)
    
    def verify_answer(self, correct_answer: str, generated_answer: str) -> bool:
        prompt = self._format_referee_prompt(correct_answer, generated_answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs["input_ids"], max_length=50, temperature=0.3)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        match = re.search(r"\\boxed\{(correct|incorrect)\}", response)
        if match:
            verdict = match.group(1)
            return verdict == "correct"
        else:
            raise ValueError("Referee model did not return a valid response.")

    def _format_referee_prompt(self, correct_answer: str, generated_answer: str) -> str:
        return (
            f"You are a referee for logical, step-by-step problem-solving.\n\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Generated Answer: {generated_answer}\n\n"
            f"Your task: Verify if the generated answer is correct. "
            "Think step-by-step and provide a final verdict: \\boxed{correct} or \\boxed{incorrect}.\n\n"
        )