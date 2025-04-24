import heapq
from idea_generation import IdeaGenerator
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
import random

MAX_REASONING_DEPTH = 5 
INITIAL_IDEAS = 1
BEAM_WIDTH = 1

class RandomBeamReasoning(ReasoningArchitecture):
    def __init__(self, reasoning_model: ReasoningModel,
                  max_reasoning_depth: int = MAX_REASONING_DEPTH, initial_ideas: int = INITIAL_IDEAS, beam_width: int = BEAM_WIDTH):

        self.reasoning_model = reasoning_model
        self.max_reasoning_depth = max_reasoning_depth
        self.initial_ideas = initial_ideas
        self.beam_width = beam_width

    def solve(self, question: str):
        """Perform beam reasoning."""
        beam = []
        self.reasoning_model.set_stop_conditions(stop_tokens=["<|im_start|><|im_end|>"], stop_strings=["\n\n", "<|im_start|>"])

        for depth in range(self.max_reasoning_depth):
            print(f"*** REASONING DEPTH {depth + 1}/{self.max_reasoning_depth} ***")
            new_beam = []
            for i, trace in enumerate(beam):
                #check if reasoning is finished for this beam
                if len(trace) > 0 and (trace[-1].endswith("<|im_start|>") or trace[-1].endswith("<|im_end|>")):
                    print(f"Beam {i + 1}/{self.beam_width} already has a final answer. Skipping...")
                    new_beam.append(trace)
                    continue
                
                print(f"\nProcessing beam {i + 1}/{self.beam_width} with previous steps: {trace}")
                for j in range(self.initial_ideas):
                    print(f"\tGenerating next reasoning step {j+1}/{self.initial_ideas}")
                    reason_prompt = self._format_prompt(question, trace) + "For the next step, "
                    reasoning_step = "For the next step, " + self.reasoning_model.generate_response(reason_prompt)
                    new_trace = trace + [reasoning_step]

                    new_beam.append(new_trace)
                
            beam = random.sample(new_beam, min(self.beam_width, len(new_beam)))

        best_trace = random.choice(beam)
        final_answer = False if best_trace[-1].endswith("<|im_start|>") else True
        prompt = self._format_prompt(question, best_trace, final_answer=final_answer)
        self.reasoning_model.set_stop_conditions(stop_tokens=["<|im_end|>"], stop_strings=[])
        final_answer = self.reasoning_model.generate_response(prompt)
        return final_answer, best_trace

    def get_total_tokens(self):
        return self.reasoning_model.get_total_tokens()
