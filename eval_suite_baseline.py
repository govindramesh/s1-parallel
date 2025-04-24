from datasets import load_dataset
import random
import argparse
import pandas as pd

from random_beam_reasoning import RandomBeamReasoning
from reasoning_architecture import ReasoningArchitecture
from reasoning_model import ReasoningModel
from sequential import SequentialReasoning

def format_mcq(question, answer_choices):
    formatted_question = question + "\n"
    for i, choice in enumerate(answer_choices):
        formatted_question += f"{chr(65 + i)}. {choice}\n"
    return formatted_question

def eval_gpqa(model: ReasoningArchitecture):
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

        correct_answer_index = answer_choices.index(correct_answer)                

        formatted_question = format_mcq(problem, answer_choices)

        solution = correct_answer
        final_answer, best_trace = model.solve(formatted_question)
        results_df = pd.concat([results_df, pd.DataFrame([{
            "problem": formatted_question,
            "solution": solution,
            "correct_letter": chr(65 + correct_answer_index),
            "final_answer": final_answer,
            "trace": best_trace,
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
    max_depths = [8]
    beam_widths = [2]
    initial_ideas = [2]
    datasets = ["gpqa"]#["aime"]

    reasoning_model = ReasoningModel(tensor_parallel_size = 4)

    # Loop through combinations and execute the command
    for max_depth in max_depths:
        for beam_width in beam_widths:
            for ideas in initial_ideas:
                for dataset in datasets:
                    print(f"Evaluating with configuration - Max depth: {max_depth}, Beam width: {beam_width}, Ideas: {ideas}, Dataset: {dataset}")

                    reasoning_model.total_tokens = 0

                    beam_reasoning = RandomBeamReasoning(
                        reasoning_model=reasoning_model,
                        max_reasoning_depth=max_depth,
                        initial_ideas=ideas,
                        beam_width=beam_width
                    )

                    
                    print("Evaluating on GPQA dataset")
                    results_df, total_tokens = eval_gpqa(beam_reasoning)

                    print(f"Total tokens used: {total_tokens}")
                    results_df.to_csv(f"eval_results_{dataset}_{max_depth}_{beam_width}_{ideas}_{total_tokens}.csv", index=False)
                    print("Evaluation results saved to file")


                    sequential = SequentialReasoning(
                        reasoning_model=reasoning_model,
                    )

                    print("Evaluating on GPQA dataset")
                    results_df, total_tokens = eval_gpqa(sequential)

                    print(f"Total tokens used: {total_tokens}")
                    results_df.to_csv(f"eval_results_{dataset}_{max_depth}_{beam_width}_{ideas}_{total_tokens}.csv", index=False)
                    print("Evaluation results saved to file")

