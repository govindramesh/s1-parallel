import re
from vllm import LLM, SamplingParams

class IdeaGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size: int = 1):
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    def generate_ideas(self, question: str, trace: list[str], num_ideas: int, max_tokens=300, fallback_temp=0.8) -> list:
        prompt = self._format_idea_prompt(question, trace, num_ideas)

        # Step 1: Try deterministic generation
        params = SamplingParams(temperature=0.3, max_tokens=max_tokens, stop=None)
        outputs = self.llm.generate(prompt, sampling_params=params)

        first_response = outputs[0].outputs[0].text
        ideas = self._parse_ideas_list(first_response)

        # Step 2: Fallback if not enough ideass
        if len(ideas) < num_ideas:
            print(f"Only {len(ideas)} ideas generated. Falling back to higher temperature...")

            fallback_params = SamplingParams(temperature=fallback_temp, max_tokens=max_tokens, stop=None, n=2, top_p=0.95)
            fallback_outputs = self.llm.generate(prompt, sampling_params=fallback_params)

            for output in fallback_outputs[0].outputs:
                ideas += self._parse_ideas_list(output.text)

            # Deduplicate and truncate
            ideas = list(dict.fromkeys(ideas))  # Preserves order

        return ideas[:num_ideas]

    def _format_idea_prompt(self, question: str, trace: str, num_ideas: int) -> str:
        trace = f'Reasoning so far: {"\n\n".join(trace)}' if trace else ''
        return f"""
Problem: {question}

{trace}

List {num_ideas} diverse, concrete ideas to explore next. Each should advance the reasoning in a different plausible direction.

Output format:
1. <idea one>
2. <idea two>
...
"""

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