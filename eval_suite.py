from datasets import load_dataset
import random
import argparse
import re
import pandas as pd

from beam_reasoning import BeamReasoning
from idea_generation import IdeaGenerator
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from reward_model import RewardModel
from referee import RefereeModel

def format_mcq(question, answer_choices):
    formatted_question = question + "\n"
    for i, choice in enumerate(answer_choices):
        formatted_question += f"{chr(65 + i)}. {choice}\n"
    return formatted_question

def extract_answer(final_answer):
    match = re.search(r"\\boxed{(.*?)}", final_answer)
    if match:
        return match.group(1)
    return None

def eval_gpqa(model: ReasoningArchitecture, referee: RefereeModel):
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    print("GPQA Dataset successfully loaded")
    results_df = pd.DataFrame(columns=["problem", "solution", "final_answer", "trace", "correct"])

    for i, row in enumerate(ds['train']):
        if i < 136:
            continue
        problem = row['Question']
        print(f"Problem: {problem}")
        correct_answer = row['Correct Answer']
        incorrect_answer_1 = row['Incorrect Answer 1']
        incorrect_answer_2 = row['Incorrect Answer 2']
        incorrect_answer_3 = row['Incorrect Answer 3']

        answer_choices = [
            correct_answer,
            incorrect_answer_1,
            incorrect_answer_2,
            incorrect_answer_3
        ]
        random.shuffle(answer_choices)

        formatted_question = format_mcq(problem, answer_choices)

        solution = correct_answer
        final_answer, best_trace = model.solve(formatted_question)
        answer = extract_answer(final_answer)
        correct = None
        try:
            if answer is not None:
                correct = referee.verify_answer(formatted_question, solution, answer)
            else:
                correct = referee.verify_answer(formatted_question, solution, final_answer)
        except ValueError as e:
            print(f"Error verifying answer: {e}")

        print("Result: " + str(correct))

        results_df = pd.concat([results_df, pd.DataFrame([{
            "problem": formatted_question,
            "solution": solution,
            "final_answer": final_answer,
            "trace": best_trace,
            "correct": correct
        }])], ignore_index=True)
        
    total_tokens = model.get_total_tokens()
    return results_df, total_tokens

def eval_aime_2024(model: ReasoningArchitecture, referee: RefereeModel):
    ds = load_dataset("Maxwell-Jia/AIME_2024")
    print("AIME Dataset successfully loaded")

    results_df = pd.DataFrame(columns=["problem", "solution", "final_answer", "trace", "correct"])

    for row in ds['train']:
        problem = row['Problem']
        final_answer, best_trace = model.solve(problem)
        model_answer = extract_answer(final_answer)
        model_answer_num = None
        correct = None

        try:
            model_answer_num = int(model_answer)
        except ValueError:
            print(f"Error converting answer to int: {model_answer}")

        correct_answer = int(row['Answer'])
        if model_answer_num == correct_answer:
            correct = True
        else:
            correct = False

        print("Result: " + str(correct))

        results_df = pd.concat([results_df, pd.DataFrame([{
            "problem": problem,
            "solution": correct_answer,
            "final_answer": final_answer,
            "trace": best_trace,
            "correct": correct
        }])], ignore_index=True)

    total_tokens = model.get_total_tokens()
    return results_df, total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth for reasoning")
    parser.add_argument("--beam_width", type=int, default=2, help="Beam width for reasoning")
    parser.add_argument("--ideas", type=int, default=2, help="Number of ideas to generate at each step")
    parser.add_argument("--dataset", type=str, default="gpqa", choices=["gpqa", "aime"], help="Dataset to evaluate on")
    args = parser.parse_args()

    print(args)

    max_depth = args.max_depth
    beam_width = args.beam_width
    ideas = args.ideas
    dataset = args.dataset

    ### ignoring args for sbatch
    max_depths = [12]
    beam_widths = [3]
    initial_ideas = [3]
    datasets = ["gpqa"]#["aime"]

    reasoning_model = ReasoningModel(tensor_parallel_size = 4)
    idea_model = IdeaGenerator(device_num = 4)
    reward_model = RewardModel(model_path='infly/Universal-PRM-7B', device_num=5) #model_path='infly/Universal-PRM-7B', 
    referee = RefereeModel(model=idea_model.model, device_num=4)

    # Loop through combinations and execute the command
    for max_depth in max_depths:
        for beam_width in beam_widths:
            for ideas in initial_ideas:
                for dataset in datasets:
                    print(f"Evaluating with configuration - Max depth: {max_depth}, Beam width: {beam_width}, Ideas: {ideas}, Dataset: {dataset}")

                    reasoning_model.total_tokens = 0

                    beam_reasoning = BeamReasoning(
                        idea_model=idea_model,
                        reasoning_model=reasoning_model,
                        reward_model=reward_model,
                        max_reasoning_depth=max_depth,
                        initial_ideas=ideas,
                        beam_width=beam_width
                    )

                    
                    if dataset == "aime":
                        print("Evaluating on AIME dataset")
                        results_df, total_tokens = eval_aime_2024(beam_reasoning, referee)
                    else:
                        print("Evaluating on GPQA dataset")
                        results_df, total_tokens = eval_gpqa(beam_reasoning, referee)

                    print(f"Total tokens used: {total_tokens}")
                    results_df.to_csv(f"eval_results_{dataset}_{max_depth}_{beam_width}_{ideas}_{total_tokens}.csv", index=False)
                    print("Evaluation results saved to file")
