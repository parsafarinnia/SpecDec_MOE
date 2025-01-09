from eagle.model.ea_model import EaModel
from eagle.model.moe_model import MOEagleModel
from fastchat.model import get_conversation_template
import torch
MOE_setting = True
num_drafts=3

model = EaModel.from_pretrained(
        base_model_path="lmsys/vicuna-7b-v1.3",
        ea_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=3,
        Moe_setting = MOE_setting,
        num_drafts = 3,
        top_k_moe = 2
    )
model.eval()
your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)

output=model.tokenizer.decode(output_ids[0]) 
print(output)
