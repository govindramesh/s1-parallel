import heapq
from idea_generation import IdeaGenerator
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from reward_model import RewardModel

MAX_REASONING_DEPTH = 5 
INITIAL_IDEAS = 5
BEAM_WIDTH = 3

class BeamReasoning(ReasoningArchitecture):
    def __init__(self, idea_model: IdeaGenerator, reasoning_model: ReasoningModel, reward_model: RewardModel):
        self.idea_model = idea_model
        self.reasoning_model = reasoning_model
        self.reward_model = reward_model

    def solve(self, question: str):
        """Perform beam reasoning."""
        initial_reward = 0.0
        beam = [(initial_reward, [])]
        heapq.heapify(beam)

        for _ in range(MAX_REASONING_DEPTH):
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

        best_trace = max(beam, key=lambda x: x[0])[1]
        prompt = self._format_prompt(question, best_trace, final_answer=True)
        final_answer = self.reasoning_model.generate_response(prompt)
        return final_answer, best_trace
