import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer (quantized for efficiency)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def format_idea_prompt(question: str, trace: str, num_ideas: int) -> str:
    return f"""
Problem: {question}

Reasoning so far:
{trace}

List {num_ideas} diverse, concrete ideas to explore next. Each should advance the reasoning in a different plausible direction.

Output format:
1. <idea one>
2. <idea two>
...
"""

def parse_ideas_list(text: str) -> list[str]:
    """
    Parses the generated ideas into a clean list
    """
    lines = text.strip().splitlines()
    ideas = []
    for line in lines:
        match = re.match(r"\s*\d+\.\s*(.*)", line)
        if match:
            ideas.append(match.group(1).strip())
    return ideas

def generate_ideas_json(question: str, trace: str, num_ideas: int, max_tokens=200, fallback_temp=0.7) -> dict:
    prompt = format_idea_prompt(question, trace, num_ideas)
    
    # Initial generation to get ideas
    output = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.3,
        do_sample=False,  # We start deterministic
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    generated = output[len(prompt):].strip()
    ideas = parse_ideas_list(generated)

    # If we don't have enough ideas, we fallback to sampling with temperature
    if len(ideas) < num_ideas:
        print(f"Only {len(ideas)} ideas generated. Falling back to temperature-based sampling to generate more.")
        output_fallback = generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=fallback_temp,  # higher temperature for diversity
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )[0]["generated_text"]

        generated_fallback = output_fallback[len(prompt):].strip()
        ideas_fallback = parse_ideas_list(generated_fallback)

        # Combine original and fallback ideas, ensuring uniqueness
        ideas = list(set(ideas + ideas_fallback))

    # Ensure the output length is exactly num_ideas
    ideas = ideas[:num_ideas]
    
    return {"ideas": ideas}


if __name__ == "__main__":
    question = "Is the number 97 a prime number?"
    trace = "We want to check if 97 has any divisors other than 1 and itself."
    num_ideas = 5  # Request 5 ideas

    ideas_json = generate_ideas_json(question, trace, num_ideas)
    print(ideas_json)
