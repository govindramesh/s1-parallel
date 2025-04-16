from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from reward_model import RewardModel

class ParallelReasoning(ReasoningArchitecture):
    def __init__(
        self, 
        reasoning_model: ReasoningModel, 
        reward_model: RewardModel, 
        num_branches: int = 5,
        min_reasoning_tokens: int = 2048,
    ):
        """
        Initialize the ParallelReasoning class.

        Args:
            reasoning_model (ReasoningModel): The model used for reasoning.
            reward_model (RewardModel): The model used for evaluating reasoning steps.
            min_reasoning_tokens (int): Minimum number of tokens for reasoning.
        """
        super().__init__()

        self.reasoning_model = reasoning_model
        reasoning_model.set_stop_tokens(['<|im_end|>'])
        self.reward_model = reward_model
        self.num_branches = num_branches
        self.min_reasoning_tokens = min_reasoning_tokens

    def solve(self, question: str):
        max_reward = float("-inf")
        best_trace = None
        best_answer = None

        for i in range(self.num_branches):
            print(f"Branch {i+1}/{self.num_branches}: Starting reasoning process.")
            prompt = self._format_prompt(question)
            
            reasoning = self.reasoning_model.generate_response(prompt, min_tokens=self.min_reasoning_tokens)
            trace = reasoning.split("\n\n")
            prompt = self._format_prompt(question, reasoning, final_answer=True)
            final_answer = self.reasoning_model.generate_response(prompt)
            reward = sum(self.reward_model.evaluate_steps(question, trace)) / len(trace)
            print(f"Calculated reward: {reward}")

            if reward > max_reward:
                print(f"New best reward found: {reward}")
                max_reward = reward
                best_trace = trace
                best_answer = final_answer

        print(f"Best answer: {best_answer}")
        print(f"Best trace: {best_trace}")
        return best_answer, best_trace