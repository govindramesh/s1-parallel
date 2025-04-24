from abc import ABC, abstractmethod
from typing import Optional

class ReasoningArchitecture(ABC):
    @abstractmethod
    def solve(self, question: str) -> tuple[str, list[str]]:
        pass

    @abstractmethod
    def get_total_tokens(self) -> int:
        pass

    def _format_prompt(
        self, 
        question: str, 
        trace: Optional[list[str]] = None, 
        idea: Optional[str] = None, 
        final_answer: bool = False,
    ) -> str:
        """
        Format the prompt for the reasoning model by including the question, 
        context from previous reasoning steps, the current idea, and a clear problem statement.

        Args:
            question (str): The main question or problem to solve.
            trace (list[str]): A list of reasoning steps taken so far.
            idea (str): The current idea to incorporate into the reasoning.
            final_answer (bool): Indicates whether the final answer is being requested. Use when reasoning is stopped early.

        Returns:
            str: A formatted prompt string for the reasoning model.
        """
        context = ''.join(trace) if trace else None
        
        prompt = (
            f"<|im_start|>system\n"
            f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
            f"Your task is to reason through problems step by step to provide accurate and logical answers."
            f"Only provide the letter for your final answer, not the full answer. Either A, B, C, or D.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}"
        )

        prompt += "<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n"

        if context:
            prompt += context

        if idea:
            prompt += "Next, " + idea + "\n"

        if final_answer:
            prompt += "<|im_start|>"

        if prompt.endswith("<|im_start|>"):
            prompt += "answer\nFinal answer:"

        return prompt
    