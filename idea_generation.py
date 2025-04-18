import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class IdeaGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", device_num: int = 0):
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.bfloat16)

    def generate_ideas(self, question: str, trace: list[str], num_ideas: int, max_tokens=300, fallback_temp=0.8) -> list:
        prompt = self._format_idea_prompt(question, trace, num_ideas)

        # Step 1: Try deterministic generation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + max_tokens,
            temperature=0.3,
            do_sample=True
        )

        first_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        ideas = self._parse_ideas_list(first_response)

        # Step 2: Fallback if not enough ideas
        if len(ideas) < num_ideas:
            print(f"Only {len(ideas)} ideas generated. Falling back to higher temperature...")

            fallback_outputs = self.model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_tokens,
                temperature=fallback_temp,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=2
            )

            for output in fallback_outputs:
                ideas += self._parse_ideas_list(self.tokenizer.decode(output, skip_special_tokens=True))

            # Deduplicate and truncate
            ideas = list(dict.fromkeys(ideas))  # Preserves order

        return ideas[:num_ideas]

    def _format_idea_prompt(self, question: str, trace: str, num_ideas: int) -> str:
        trace = 'Reasoning so far: ' + '\n'.join(trace) if trace else 'Reasoning so far: None'
        return f"""You are an idea generator for logical, step-by-step problem-solving.

Problem: {question}

{trace}


Your task: provide {num_ideas} concrete ideas of what the next step should be. These ideas should advance any provided reasoning and aim to solve the problem.

Output your ideas in a numbered list, without any additional text."""


    def _parse_ideas_list(self, text: str) -> list[str]:
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