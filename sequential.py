from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel

class SequentialReasoning(ReasoningArchitecture):
    def __init__(self, reasoning_model: ReasoningModel):
        """
        Initialize the SequentialReasoning class.

        Args:
            reasoning_model (ReasoningModel): The model used for reasoning.
        """
        super().__init__()
        self.reasoning_model = reasoning_model
        self.reasoning_model.set_stop_conditions(stop_tokens=["<|im_start|><|im_end|>"], stop_strings=["<|im_start|>"])    

    def solve(self, question: str):
        """
        Perform sequential reasoning.

        Args:
            question (str): The question to be solved.

        Returns:
            str: The final answer.
            list[str]: The reasoning trace.
        """
        prompt = self._format_prompt(question)
        reasoning = self.reasoning_model.generate_response(prompt)
        trace = reasoning.split("\n\n")
        prompt = self._format_prompt(question, reasoning, final_answer=True)
        self.reasoning_model.set_stop_conditions(stop_tokens=["<im_end|>"], stop_strings=[])
        final_answer = self.reasoning_model.generate_response(prompt)
        
        return final_answer, trace
    
    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens used in the reasoning process.

        Returns:
            int: The total number of tokens.
        """
        return self.reasoning_model.get_total_tokens()