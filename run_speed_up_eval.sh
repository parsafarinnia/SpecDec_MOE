
CUDA_VISIBLE_DEVICES=2 python3 -m eagle.evaluation.gen_ea_answer_vicuna --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3  --model-id "EAGLE_MOE_3EXP_TOP1" 

# CUDA_VISIBLE_DEVICES=2 python3 -m eagle.evaluation.gen_baseline_answer_vicuna --model-path lmsys/vicuna-7b-v1.3 --model-id baseline_vicuna