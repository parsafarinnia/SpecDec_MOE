from eagle.model.ea_model import EaModel
from eagle.model.moe_model import MOEagleModel
from fastchat.model import get_conversation_template
import torch
MOE_setting = True
num_drafts=3

import torch
import torch.nn as nn

def count_parameters(model):
  """Counts the total number of parameters in a PyTorch model.

  Args:
    model: A PyTorch model (instance of nn.Module).

  Returns:
    A tuple containing:
      - total_params: The total number of parameters in the model.
      - trainable_params: The number of trainable parameters in the model 
                         (i.e., parameters that require gradients).
  """
  total_params = 0
  trainable_params = 0
  for name, parameter in model.named_parameters():
    num_params = parameter.numel()
    total_params += num_params
    if parameter.requires_grad:
      trainable_params += num_params
    # print(f"Layer: {name}, Parameters: {num_params}, Trainable: {parameter.requires_grad}")  # Optional: Print details for each layer

  print(f"Total parameters: {total_params}")
  print(f"Trainable parameters: {trainable_params}")
  print(f"Non-trainable parameters: {total_params - trainable_params}")

  return total_params, trainable_params

from huggingface_hub import hf_hub_download
ea_model_path = "yuhuili/EAGLE-Vicuna-7B-v1.3"
load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
ea_layer_state_dict = torch.load(load_model_path)
# if not MOE_setting:

model= EaModel.from_pretrained(
            base_model_path="lmsys/vicuna-7b-v1.3",
            ea_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=10,
            Moe_setting = True,
            num_drafts = 3,
            top_k_moe = 2
        )
# else:
# model_normal = EaModel.from_pretrained(
#             base_model_path="lmsys/vicuna-7b-v1.3",
#             ea_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             device_map="auto",
#             total_token=10,
#             Moe_setting = False,
#             num_drafts = 3,
#             top_k_moe = 2
#         )
# count_parameters(model_MOE)
# print("model MOE")
# count_parameters(model_normal)
# print("model eagle")
# '''
# python -m eagle.evaluation.gen_ea_answer_vicuna\
# 		 --ea-model-path "yuhuili/EAGLE-Vicuna-7B-v1.3"\ 
# 		 --base-model-path "lmsys/vicuna-7b-v1.3"\
# '''
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


