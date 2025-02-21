#!/bin/bash
#
# run_train_moe.sh
#
# This script will:
#   1) Dynamically assign MASTER_ADDR & MASTER_PORT
#   2) Set environment variables
#   3) Launch accelerate with the same arguments as in launch.json
#

# 1) Create logs directory if it doesn't exist:
mkdir -p logs

# 1) Dynamically set the environment:
export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDZV_ID=$RANDOM
echo "RDZV Endpoint is: $MASTER_ADDR:$MASTER_PORT (ID=$RDZV_ID)"
run_name="train_moe_top2k_4exp_with_jitter_5epoch_node10"
# 2) Other environment variables:
export CUDA_VISIBLE_DEVICES="" 
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="online"

accelerate launch -m \
    --main_process_port $MASTER_PORT \
    --main_process_ip $MASTER_ADDR \
    --rdzv_backend static \
    --mixed_precision=bf16 \
    eagle.train.main \
    --basepath /work/farinneya/models/Llama_3.1_8B_Instruct \
    --tmpdir /work/farinneya/eagle_data_llama3_8B/sharegpt_0_67999_mufp16 \
    --cpdir /work/farinneya/checkpoints/eagle_experiments/share_gpt_MOE_Eeagle_test_${run_name} \
    --configpath /home/farinneya/SpecDec_MOE/eagle/train/model_configs/LLaMA3.1-Instruct-8B.json\
    --train-config-file /home/farinneya/SpecDec_MOE/eagle/train/training_configs/moe_config.json\
    --gradient-accumulation-steps 1 \
    --bs 4\
    --run_name ${run_name} > logs/${run_name}_logs.txt 2>&1





# #!/bin/bash

# export MASTER_ADDR="$(hostname --fqdn)"
# export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
# export RDVZ_ID=$RANDOM
# echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

# run_name="original_eagle_v3_two_layer_expansion_2"

# NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES="4,5,7" \
# TOKENIZERS_PARALLELISM=false WANDB_MODE=offline accelerate launch -m \
#     --main_process_port $MASTER_PORT \
#     --main_process_ip $MASTER_ADDR \
#     --rdzv_backend static \
#     --mixed_precision=bf16 \
#     eagle.train.main \
#     --basepath /work/saeed-data/model-weights/Llama-3.1-8B-Instruct \
#     --tmpdir /work/saeed-data/datasets/eagle_data/sharegpt_0_67999_mufp16 \
#     --cpdir /work/saeed-data/checkpoints/eagle_experiments/share_gpt_${run_name} \
#     --configpath /home/saeednajafi/EAGLE/eagle/train/EAGLE-LLaMA3-Instruct-8B \
#     --num_hidden_layers 2 \
#     --expansion_factor 2 \
#     --add_next_token_loss no \
#     --save_to_hf no \
#     --train_lm_head_em_table no \
#     --gradient-accumulation-steps 2 \
#     --include_top_k_loss no \
#     --topk 5 \
#     --bs 2 \
#     --run_name ${run_name} > logs/${run_name}_logs.txt 2>&1