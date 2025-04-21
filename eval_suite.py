from datasets import load_dataset

from beam_reasoning import BeamReasoning
from idea_generation import IdeaGenerator
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from reward_model import RewardModel
from referee import RefereeModel
import argparse
import re
import pandas as pd

def extract_answer(final_answer):
    match = re.search(r"\\boxed{(.*?)}", final_answer)
    if match:
        return match.group(1)
    return None

def eval_gpqa(model: ReasoningArchitecture, referee: RefereeModel):
    ds = load_dataset("hendrydong/gpqa_diamond")
    print("Dataset successfully loaded")
    results_df = pd.DataFrame(columns=["problem", "solution", "final_answer", "correct"])

    iteration_count = 0
    for row in ds['test']:
        if iteration_count >= 2:
            break
        problem = row['problem']
        print(f"Problem: {problem}")
        solution = row['solution']
        final_answer, best_trace = model.solve(problem)
        answer = extract_answer(final_answer)
        correct = None
        try:
            correct = referee.verify_answer(problem, solution, answer)
        except ValueError as e:
            print(f"Error verifying answer: {e}")
            continue

        results_df = results_df.append({
            "problem": problem,
            "solution": solution,
            "final_answer": final_answer,
            "trace": best_trace,
            "correct": correct
        }, ignore_index=True)
        
        iteration_count += 1

    total_tokens = model.get_total_tokens()

    return results_df, total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth for reasoning")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for reasoning")
    args = parser.parse_known_args()

    idea_model = IdeaGenerator(device_num = 4)
    reasoning_model = ReasoningModel(tensor_parallel_size = 4)
    reward_model = RewardModel(device_num=5)

    beam_reasoning = BeamReasoning(
        idea_model=idea_model,
        reasoning_model=reasoning_model,
        reward_model=reward_model,
        max_reasoning_depth=args.max_depth,
        initial_ideas=3,
        beam_width=args.beam_width
    )

    referee = RefereeModel()

    results_df, total_tokens = eval_gpqa(beam_reasoning, referee)

    print(f"Total tokens used: {total_tokens}")
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")

