from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/parscratch/users/acs23sl/dataset/LLaVA-NeXT-Data/llava_next_raw_format/sharegpt4v/wikiart/canaletto_london-the-old-horse-guards-from-st-james-s-park-1749.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
{'role': 'assistant', 'content': "The image is a captivating oil painting that transports us back to the 18th century. It beautifully captures a cityscape, specifically a representation of the city of Paris. The painting is executed in "}
]

# [{'role': 'user', 'content': [{'type': 'image', 'image': '/mnt/parscr
# atch/users/acs23sl/dataset/LLaVA-NeXT-Data/llava_next_raw_format/sharegpt4v/wikiart/canaletto_london-the-old-horse-guards-from-st-james-s-park-1749.jpg'},
# {'type': 'text', 'text': 'Describe this image in detail.'}]},
# {'role': 'assistant', 'content': "The image is a captivating oil painting that transports us back to the 18th century. It beautifully captures a cityscape, specifically a representation of the city of Paris. The painting is executed in the Rococo style, a movement known for its intricate details, ornate embellishments, and curved, asymmetrical forms.\n\nThe scene unfolds in a large open square bustling with life. People and carriages animate the scene, adding a sense of dynamism and vitality. The buildings, rendered in the Baroque style, command attention with their grandeur and architectural complexity. This style is characterized by extravagant details, dramatic use of light, and a sense of movement within the structure.\n\nThe color palette is dominated by earthy tones of brown and green, lending a sense of warmth and familiarity to the scene. These colors are beautifully contrasted by the sky, which is painted in shades of blue with white clouds scattered across it, adding depth and dimension to the painting.\n\nThe painting is presented in a landscape format, allowing the artist to capture the breadth and scale of the cityscape. Despite the historical context, the painting feels timeless, offering a glimpse into the past while still resonating with contemporary viewers. It's a testament to the enduring power of art to capture and convey the spirit of a place and time."}]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print('###', text)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
print('inputs', inputs)
print('model.config', model.config)
for name, module in model.named_modules():
    print(name)
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
