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
<<<<<<< HEAD
run_name="train_moe_test_3exp_2topk_3epoch_node10"
date_run="March_3rd"
=======
run_name="train_moe_top2k_3exp_with_jitter_3epoch_node10_for_Llama3.1_12PM_3March"
>>>>>>> 1e384bd7e8a71e08d3c437d8d5a94a5c2bbc5d82
# 2) Other environment variables:
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" 
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="online"


accelerate launch -m \
    --main_process_port $MASTER_PORT \
    --main_process_ip $MASTER_ADDR \
    --rdzv_backend static \
    --mixed_precision=bf16 \
    eagle.train.main \
<<<<<<< HEAD
    --basepath /work/farinneya/models/Llama_3.1_8B_Instruct \
    --tmpdir /work/farinneya/eagle_data_llama3_8B/sharegpt_0_67999_mufp16 \
    --cpdir /work/farinneya/checkpoints/eagle_experiments/${date_run}_share_gpt_MOE_Eeagle_test_${run_name} \
=======
    --basepath /work/farinneya/farinneya/models/Llama_3.1_8B_Instruct/ \
    --tmpdir /work/farinneya/farinneya/eagle_data_llama3_8B/sharegpt_0_67999_mufp16 \
    --cpdir /work//farinneya/farinneya/checkpoints/eagle_experiments/share_gpt_MOE_Eeagle_test_${run_name} \
>>>>>>> 1e384bd7e8a71e08d3c437d8d5a94a5c2bbc5d82
    --configpath /home/farinneya/SpecDec_MOE/eagle/train/model_configs/LLaMA3.1-Instruct-8B.json\
    --train-config-file /home/farinneya/SpecDec_MOE/eagle/train/training_configs/moe_config.json\
    --configpath-moe /home/farinneya/SpecDec_MOE/eagle/train/model_congifs/llama3.1_8B_instruct_moe_config.json\
    --gradient-accumulation-steps 1 \
    --bs 4\
    --run_name ${run_name} > logs/${run_name}_logs.txt 2>&1

