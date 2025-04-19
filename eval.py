from datasets import load_dataset

from beam_reasoning import BeamReasoning
from idea_generation import IdeaGenerator
from reasoning_model import ReasoningModel
from reward_model import RewardModel
import argparse
import re

ds = load_dataset("hendrydong/gpqa_diamond")

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth for reasoning")
parser.add_argument("--beam_width", type=int, default=5, help="Beam width for reasoning")
args = parser.parse_args()


idea_model = IdeaGenerator()
reasoning_model = ReasoningModel()
reward_model = RewardModel()

beam_reasoning = BeamReasoning(
    idea_model=idea_model,
    reasoning_model=reasoning_model,
    reward_model=reward_model,
    max_reasoning_depth=args.max_depth,
    initial_ideas=3,
    beam_width=args.beam_width
)

def extract_answer(final_answer):
    match = re.search(r"\\boxed{(.*?)}", final_answer)
    if match:
        return match.group(1)
    return None

for row in ds['test']:
    problem = row['problem']
    solution = row['solution']

    final_answer, best_trace = beam_reasoning.solve(problem)

    answer = extract_answer(final_answer)

total_tokens = idea_model.get_total_tokens() + reasoning_model.get_total_tokens()

