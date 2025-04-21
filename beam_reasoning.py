import heapq
from idea_generation import IdeaGenerator
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from reward_model import RewardModel

MAX_REASONING_DEPTH = 5 
INITIAL_IDEAS = 1
BEAM_WIDTH = 1

class BeamReasoning(ReasoningArchitecture):
    def __init__(self, idea_model: IdeaGenerator, reasoning_model: ReasoningModel, reward_model: RewardModel,
                  max_reasoning_depth: int = MAX_REASONING_DEPTH, initial_ideas: int = INITIAL_IDEAS, beam_width: int = BEAM_WIDTH):

        self.idea_model = idea_model
        self.reasoning_model = reasoning_model
        self.reward_model = reward_model
        self.max_reasoning_depth = max_reasoning_depth
        self.initial_ideas = initial_ideas
        self.beam_width = beam_width

    def solve(self, question: str):
        """Perform beam reasoning."""
        initial_reward = 0.0
        beam = [(initial_reward, [])]
        heapq.heapify(beam)

        for depth in range(self.max_reasoning_depth):
            print(f"*** REASONING DEPTH {depth + 1}/{self.max_reasoning_depth} ***")
            new_beam = []
            for i, trace in enumerate(heapq.nlargest(self.beam_width, beam)):
                #check if reasoning is finished for this beam
                if len(trace[1]) > 0 and (trace[1][-1].endswith("<|im_start|>") or trace[1][-1].endswith("<|im_end|>")):
                    print(f"Beam {i + 1}/{self.beam_width} already has a final answer. Skipping...")
                    new_beam.append(trace)
                    continue
                
                trace = trace[1] # Extract the trace from the tuple (reward, trace)
                print(f"\nProcessing beam {i + 1}/{self.beam_width} with previous steps: {trace}")
                new_ideas = self.idea_model.generate_ideas(question, trace, self.initial_ideas)
                for j, idea in enumerate(new_ideas):
                    print(f"\tGenerating reasoning for idea {j+1}/{self.initial_ideas}: {idea}")
                    reason_prompt = self._format_prompt(question, trace, idea)
                    reasoning_step = "Next, " + idea + "\n" + self.reasoning_model.generate_response(reason_prompt)
                    new_trace = trace + [reasoning_step]
 
                    #need to average or take last reward
                    reward = self.reward_model.evaluate_steps(question, new_trace)[-1]
                    print(f"\tReward for new idea: {reward}\n")
                    new_beam.append((reward, new_trace))
                
            beam = heapq.nlargest(self.beam_width, new_beam, key=lambda x: x[0])

        best_trace = max(beam, key=lambda x: x[0])[1]
        final_answer = False if best_trace[-1].endswith("<|im_start|>") else True
        prompt = self._format_prompt(question, best_trace, final_answer=final_answer)
        self.reasoning_model.set_stop_conditions(stop_tokens=["<|im_end|>"], stop_strings=[])
        final_answer = self.reasoning_model.generate_response(prompt)
        return final_answer, best_trace

    def get_total_tokens(self):
        return self.idea_model.get_total_tokens() + self.reasoning_model.get_total_tokens()
