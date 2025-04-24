import re
from typing import Optional
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
import torch

class RefereeModel:
    def __init__(self, model: Optional[PreTrainedModel], model_name: str = "google/gemma-2-9b-it", device_num: int = 0):
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.bfloat16)
    
    def verify_answer(self, question: str, correct_answer: str, generated_answer: str) -> bool:
        prompt = self._format_referee_prompt(question, correct_answer, generated_answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs["input_ids"], temperature=0.3, do_sample=True, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        match = re.search(r"\\boxed\{(correct|incorrect)\}", response)
        if match:
            verdict = match.group(1)
            return verdict == "correct"
        else:
            raise ValueError("Referee model did not return a valid response.")

    def _format_referee_prompt(self, question: str, correct_answer: str, generated_answer: str) -> str:
        return (
            f"You are a referee for evaluating answers for correctness.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Generated Answer: {generated_answer}\n\n"
            f"Your task: Verify if the generated answer and correct answer are logically equivalent with respect to the question."
            "Briefly think step-by-step and provide a final verdict: \\boxed{correct} or \\boxed{incorrect}.\n\n"
        )