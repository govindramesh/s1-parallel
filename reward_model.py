# see https://huggingface.co/infly/Universal-PRM-7B, https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

class RewardModel:
    def __init__(self, model_path: str = 'Qwen/Qwen2.5-Math-PRM-7B', device_num: int = 0):
        self.model_path = model_path
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def evaluate_steps(self, question: str, trace: list[str]) -> list[float]:
        try:
            with torch.no_grad():
                if self.model_path == 'infly/Universal-PRM-7B':
                    question_wgt = question + '\n\n###\n\nThe reference answer is: There is no reference answer for this question.'
                    judge_list = []
                    
                    for step_idx in range(1, len(trace) + 1):
                        responses = ''.join(trace[:step_idx])
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
                
                if self.model_path == 'Qwen/Qwen2.5-Math-PRM-7B':
                    system_str = "Your task is to reason through problems step by step to provide accurate and logical answers."
                    concat_trace = [(step[:-2] if step.endswith("\n\n") else step) for step in trace]

                    if concat_trace[-1].endswith("<|im_start|>"):
                        concat_trace[-1] = concat_trace[-1][:-len("<|im_start|>")]

                    messages = [
                        {"role": "system", "content": system_str},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": "<extra_0>".join(concat_trace) + "<extra_0>"},
                    ]
                    conversation_str = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )

                    input_ids = self.tokenizer.encode(
                        conversation_str, 
                        return_tensors="pt", 
                    ).to(self.device)

                    outputs = self.model(input_ids=input_ids)

                    step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                    token_masks = (input_ids == step_sep_id)
                    step_reward = self._make_step_rewards(outputs[0], token_masks)
                    return step_reward[0]
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM Error in evaluate_steps: {e}")
            torch.cuda.empty_cache()
            return [-1.0]
        except Exception as e:
            print(f"Error in evaluate_steps: {e}")
            return [-1.0]
        
    def _make_step_rewards(self, logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
