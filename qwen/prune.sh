#!/bin/bash

sparsity_ratio=0.6
# ["unstructured", "4:8", "2:4"]
sparsity_type="unstructured"

# ["qwen2.5-vl-7b", "qwen2-vl-7b"]
model="qwen2-vl-7b"
# calibration data directory
data_dir="your_directory/LLaVA-NeXT-Data"
calibration_source="sharegpt4v"
# Saved results directory
res_dir="results"

random_seed=21

# {'wanda'; 'tamp': TAMP, the ACL paper, we only do uniform pruning, i.e., AMIA only; 'atv': our ATV-Pruning}
prune_method="atv"
# alpha parameter for ATV-Pruning
alpha=1

python -m qwen.prune \
    --data_path ${data_dir}/llava_next_raw_format/llava_next_raw_format_processed.json \
    --task_split_path ${data_dir}/llava_next_raw_format/task_split.json \
    --task_name $calibration_source \
    --image_folder ${data_dir}/llava_next_raw_format \
    --seed $random_seed \
    --nsamples 128 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $sparsity_type \
    --prune_method $prune_method \
    --save log \
    --save_model ${res_dir}/checkpoints/${model}_${prune_method}_${alpha}_${sparsity_ratio}_${sparsity_type}_${random_seed} \
    --model ${model} \
    --alpha ${alpha} \
