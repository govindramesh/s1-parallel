from abc import ABC, abstractmethod
from typing import Optional

class ReasoningArchitecture(ABC):
    @abstractmethod
    def solve(self, question: str) -> tuple[str, list[str]]:
        pass

    def _format_prompt(
        self, 
        question: str, 
        trace: Optional[list[str]] = None, 
        idea: Optional[str] = None, 
        final_answer: bool = False
    ) -> str:
        """
        Format the prompt for the reasoning model by including the question, 
        context from previous reasoning steps, the current idea, and a clear problem statement.

        Args:
            question (str): The main question or problem to solve.
            trace (list[str]): A list of reasoning steps taken so far.
            idea (str): The current idea to incorporate into the reasoning.
            final_answer (bool): Whether to indicate that the final answer is being requested.

        Returns:
            str: A formatted prompt string for the reasoning model.
        """
        context = "\n\n".join(trace) if trace else None
        
        prompt = (
            f"<|im_start|>system\n"
            f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            f"Your task is to reason through problems step by step to provide accurate and logical answers.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}\n"
        )

        prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n"

        if context:
            prompt += context

        if idea:
            prompt += "\n\nNext, " + idea + "\n"

        if final_answer:
            prompt += "</think>\n\n"

        return prompt
    