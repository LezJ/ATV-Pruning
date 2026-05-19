import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info


def convert_sharegpt4v_to_qwen2vl(sample, image_folder):
    """
    Convert a ShareGPT4V-style sample into Qwen2-VL message format.
    """
    image_path = os.path.join(image_folder, sample["image"])
    conversations = sample["conversations"]
    messages = []
    for turn in conversations:
        role = 'user' if turn["from"] == 'human' else 'assistant'

        if role == "user":
            assert '<image>\n' in turn['value']
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image_path},
                        {'type': 'text', 'text': turn['value'].replace('<image>\n', '')}
                    ]
                }
            )
        else:
            messages.append(
                {
                    'role': 'assistant',
                    'content': turn['value']
                }
            )

    return messages


class LazySupervisedDataset(Dataset):
    def __init__(self, processor, data_path: str, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []

        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                self.list_data_dict = json.load(f)
        else:
            raise ValueError("Only JSON dataset supported in this simplified version.")

        task_split_dict = json.load(open(data_args.task_split_path, 'r'))
        task_start_id, task_end_id = task_split_dict[data_args.task_name]
        if data_args.sample_select == 'random':
            sampled_indices = np.random.choice(
                np.arange(task_start_id, task_end_id),
                size=data_args.nsamples,
                replace=False
            )
        elif data_args.sample_select == 'longest':
            lang_length_list = []
            for i in range(task_start_id, task_end_id):
                total_lang_len = np.array([
                    len(conv["value"]) for conv in self.list_data_dict[i]["conversations"]
                ])
                lang_length_list.append(total_lang_len.sum())
            sampled_indices = np.argsort(lang_length_list)[-data_args.nsamples:]
            sampled_indices += task_start_id

        self.list_data_dict = [self.list_data_dict[idx] for idx in sampled_indices]

        self.processor = processor
        self.image_folder = data_args.image_folder

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sample = self.list_data_dict[i]

        messages = convert_sharegpt4v_to_qwen2vl(sample, self.image_folder)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # remove the extra batch dimension for each tensor
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                inputs[k] = v.squeeze(0)

        return inputs


def create_qwen2vl_dataloader(processor, data_args, num_workers=4):
    dataset = LazySupervisedDataset(processor, data_args.data_path, data_args)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, shuffle=False,
    )
    return dataloader


if __name__ == "__main__":
    class Args:
        data_path = "your_directory/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json"
        task_split_path = "your_directory/LLaVA-NeXT-Data/llava_next_raw_format/task_split.json"
        task_name = "sharegpt4v"
        image_folder = "your_directory/LLaVA-NeXT-Data/llava_next_raw_format"
        nsamples = 128
        sample_select = "random"

    data_args = Args()
    loader = create_qwen2vl_dataloader(data_args)
    for batch in loader:
        # print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        batch
        # print('23424234234')
        # break
