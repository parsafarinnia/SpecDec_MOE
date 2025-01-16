mkdir -p logs
export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
run_name="MOE_Eeagle_test"
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES="2,3" \
TOKENIZERS_PARALLELISM=false WANDB_MODE=offline accelerate launch -m \
    --main_process_port $MASTER_PORT \
    --main_process_ip $MASTER_ADDR \
    --rdzv_backend static \
    --mixed_precision=bf16 \
    eagle.train.main \
    --basepath lmsys/vicuna-7b-v1.3 \
    --tmpdir /work/parsa-data/eagle_data/sharegpt_0_67999_mufp16 \
    --cpdir /work/parsa-data/checkpoints/eagle_experiments/share_gpt_${run_name} \
    --configpath /home/farinneya/eagle/eagle/train/vicuna_7B_config.json \
    --gradient-accumulation-steps 2 \
    --bs 2 \
    --run_name ${run_name} > logs/${run_name}_logs.txt 2>&1