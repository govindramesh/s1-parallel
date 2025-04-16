# see https://huggingface.co/infly/Universal-PRM-7B
import re
from transformers import AutoModel, AutoTokenizer
import torch
import json

class RewardModel:
    def __init__(self, model_path: str = 'infly/Universal-PRM-7B', device_num: int = 0):
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def evaluate_steps(self, question: str, trace: list[str]) -> list[float]:
        question_wgt = question + '\n\n###\n\nThe reference answer is: There is no reference answer for this question.'
        judge_list = []
        
        with torch.no_grad():
            for step_idx in range(1, len(trace) + 1):
                responses = "\n\n".join(trace[:step_idx]) + "\n\n"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question_wgt}
                ]
                query_id = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True
                )
                answer_tokens = self.tokenizer(responses)['input_ids']
                answer_tokens += [self.tokenizer.eos_token_id]
                QA_ids = query_id + answer_tokens
                
                input_ids = torch.tensor([QA_ids]).long().cuda().contiguous().to(self.device)
                outputs = self.model(input_ids=input_ids)
                reward = torch.sigmoid(outputs[0]).cpu().item()
                judge_list.append(reward)
                
        return judge_list