import heapq
from idea_generation import IdeaGenerator
from reasoning_model import ReasoningModel
from reward_model import RewardModel

MAX_REASONING_DEPTH = 5 
INITIAL_IDEAS = 5
BEAM_WIDTH = 3

class BeamReasoning:
    def __init__(self, idea_model: IdeaGenerator, reasoning_model: ReasoningModel, reward_model: RewardModel):
        self.idea_model = idea_model
        self.reasoning_model = reasoning_model
        self.reward_model = reward_model

    def beam_search(self, question):
        """Perform beam reasoning."""
        initial_reward = 0.0
        beam = [(initial_reward, [])]
        heapq.heapify(beam)

        for depth in range(MAX_REASONING_DEPTH):
            print(f"Depth {depth + 1}/{MAX_REASONING_DEPTH}")
            new_beam = []
            for i, trace in enumerate(heapq.nlargest(BEAM_WIDTH, beam)):
                print(f"Processing beam {i + 1}/{BEAM_WIDTH} with previous steps: {trace[1]}")
                trace = trace[1]
                new_ideas = self.idea_model.generate_ideas(question, trace, INITIAL_IDEAS)
                for idea in new_ideas:
                    print(f"Generating reasoning step for idea: {idea}")
                    reason_prompt = self._format_prompt(question, trace, idea)
                    reasoning_step = self.reasoning_model.generate_response(reason_prompt)
                    new_trace = trace + [reasoning_step]
 
                    #need to average or take last reward
                    reward = self.reward_model.evaluate_steps(question, new_trace)[-1]
                    print(f"Reward for new trace: {reward}")
                    new_beam.append((reward, new_trace))
                
            beam = heapq.nlargest(BEAM_WIDTH, new_beam, key=lambda x: x[0])

        return [trace for _, trace in beam]

    def _format_prompt(self, question: str, trace: list[str], idea: str) -> str:
        """
        Format the prompt for the reasoning model by including the question, 
        context from previous reasoning steps, the current idea, and a clear problem statement.

        Args:
            question (str): The main question or problem to solve.
            trace (list[str]): A list of reasoning steps taken so far.
            idea (str): The current idea to incorporate into the reasoning.

        Returns:
            str: A formatted prompt string for the reasoning model.
        """
        context = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(trace))
        
        prompt = (
            f"<|im_start|>system\n"
            f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            f"Your task is to reason through problems step by step to provide accurate and logical answers.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Problem: {question}\n"
        )
        
        if context:
            prompt += f"Context:\n{context}\n"
        
        prompt += f"Idea for next reasoning step: {idea}\n"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
        return prompt
