# see https://huggingface.co/infly/Universal-PRM-7B

from transformers import AutoModel, AutoTokenizer
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'infly/Universal-PRM-7B'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(
    model_path, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

question = "It's April, and Mrs. Rylan has been busy on her farm planting different types of vegetables for the season. She has bought 20 packets of tomato seeds and 80 packets of celery seeds to plant. If a packet of tomato seeds costs $40 and a packet of celery seeds costs $30, how much money did she use to buy the seeds?"
ground_truth_solution = "The total amount of money she used to buy the tomato seeds is 20 packets * $40/packet = $<<20*40=800>>800\nThe celery seeds cost her 80 packets * $30/packet = $<<80*30=2400>>2400\nFor the seeds, Mrs. Rylan paid $2400 + $800 = $<<2400+800=3200>>3200\n#### 3200"
steps = ["To find out how much money Mrs. Rylan used to buy the seeds, we need to calculate the total cost of tomato seeds and celery seeds separately, then add them together.", "First, calculate the total cost of tomato seeds. Number of packets of tomato seeds = 20. Cost per packet of tomato seeds = $40. Total cost of tomato seeds = Number of packets of tomato seeds * Cost per packet of tomato seeds = 20 * $40 = $800.", "Second, calculate the total cost of celery seeds. Number of packets of celery seeds = 80. Cost per packet of celery seeds = $30. Total cost of celery seeds = Number of packets of celery seeds * Cost per packet of celery seeds = 80 * $30 = $2400.", "Finally, calculate the total amount of money used to buy the seeds. Total amount of money = Total cost of tomato seeds + Total cost of celery seeds = $800 + $2400 = $3200.", "Therefore, Mrs. Rylan used \\boxed{$3200} to buy the seeds."]

if ground_truth_solution != '':
    question_wgt = question + '\n\n###\n\nThe reference answer is: ' + ground_truth_solution
else:
    question_wgt = question + '\n\n###\n\nThe reference answer is: There is no reference answer for this question.'

judge_list_infer = []
with torch.no_grad():
    for step_idx in range(1, len(steps) + 1):
        responses = "\n\n".join(steps[:step_idx]) + "\n\n"
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_wgt}
            ]
        query_id = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
        answer_tokens = tokenizer(responses)['input_ids']
        answer_tokens += [tokenizer.eos_token_id]
        QA_ids = query_id + answer_tokens
        
        input_ids = torch.tensor([QA_ids]).long().cuda().contiguous()

        outputs = model(input_ids=input_ids)
        reward = torch.sigmoid(outputs[0]).cpu().item()
        judge_list_infer.append(reward)

print(judge_list_infer)     # [0.73828125, 0.7265625, 0.73046875, 0.73828125, 0.734375]
