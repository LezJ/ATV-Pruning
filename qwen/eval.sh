#!/bin/bash

tasks="ok_vqa_val2014,vizwiz_vqa_val,textvqa_val,gqa,pope,scienceqa_img,mmbench_en_dev,mme,mmmu_val"
# model_path="Qwen/Qwen2-VL-7B-Instruct"
model_path="results/checkpoints/qwen2-vl-7b_atv_1_0.6_unstructured_21"

log_dir="results"
run_name="Qwen2-VL-7B"
model="qwen2_vl"

python3 -m lmms_eval \
    --model ${model} \
    --model_args pretrained=${model_path} \
    --tasks ${tasks} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${run_name} \
    --output_path ./${log_dir}/${run_name}
