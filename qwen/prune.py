import argparse
import os
import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print('# of gpus: ', torch.cuda.device_count())

from importlib.metadata import version
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen.data_loader import create_qwen2vl_dataloader

from qwen.activation_aware_pruner import ActivationAwarePruner


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='LVLM model')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--task_split_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.') 
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str)
    parser.add_argument('--alpha', type=float, default=None, help='alpha parameter for ATV-Pruning')
    parser.add_argument("--sample_select", default="random", type=str )
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--model', type=str, choices=["qwen2.5-vl-7b", "qwen2-vl-7b"])

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.llm_sparsity_ratio == 0.5 or args.vit_sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Model & processor loading
    print(args.model)
    if args.model == 'qwen2.5-vl-7b':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    elif args.model == 'qwen2-vl-7b':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2-VL-7B-Instruct', torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    else:
        raise NotImplementedError

    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    args.is_multimodal = True

    # Dataloader
    # System prompt for LVLM

    dataloader = create_qwen2vl_dataloader(processor, data_args=args)

    assert args.sparsity_ratio != 0
    print("pruning starts")
    if args.prune_method in ['wanda', 'tamp', 'atv']:
        pruner = ActivationAwarePruner(
            model=model,
            data_loader=dataloader,
            sparsity_ratio=args.sparsity_ratio,
            pruning_method=args.prune_method,
            num_samples=args.nsamples,
            prune_n=prune_n,
            prune_m=prune_m,
            # locate llm backbone in the LVLM
            model_prefix='language_model',
            module_to_process='language_model.layers',
            alpha=args.alpha,
            )
    else:
        raise NotImplementedError

    model = pruner.prune()

    pruned_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
    print(f"Remaining Proportion: {pruned_total_size / orig_total_size * 100}%")

    # Save pruned model
    if args.save_model:
        model.save_pretrained(args.save_model)
        processor.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
