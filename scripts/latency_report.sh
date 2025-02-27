# #!/bin/bash
# cd /home/farinneya/SpecDec_MOE
# export CUDA_VISIBLE_DEVICES="6,7" 
# nsys profile --trace=cuda,osrt -o /home/farinneya/SpecDec_MOE/logs/latency_report \
# python -m eagle.evaluation.gen_ea_answer_llama3chat \
#   --ea-model-path /work/farinneya/checkpoints/eagle_experiments/share_gpt_MOE_Eeagle_test_train_moe_top2k_8exp_with_jitter_21epoch_node4/state_1 \
#   --base-model-path /work/farinneya/models/Llama_3.1_8B_Instruct \
#   > /home/farinneya/SpecDec_MOE/logs/latency_report.log 2>&1



#!/bin/bash
cd /home/farinneya/SpecDec_MOE
export CUDA_VISIBLE_DEVICES="6,7"

# Start the Python script with debugpy
python -m debugpy --listen 5678 --wait-for-client -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path /work/farinneya/checkpoints/eagle_experiments/share_gpt_MOE_Eeagle_test_train_moe_top2k_8exp_with_jitter_21epoch_node4/state_1 \
  --base-model-path /work/farinneya/models/Llama_3.1_8B_Instruct \
  > /home/farinneya/SpecDec_MOE/logs/latency_report.log 2>&1
