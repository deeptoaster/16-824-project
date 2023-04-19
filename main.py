#!/usr/bin/python
import accelerate
from argparse import Namespace
from diffusers import AutoencoderKL, UNet2DConditionModel
from io import BytesIO
import itertools
from pathlib import Path
from PIL import Image
import requests
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from train import train


# `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
# Add here the URLs to the images of the concept you are adding. 3-5 should be fine
urls = [
    "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
]


save_path = Path("./my_concept")
save_path.mkdir(exist_ok=True)
for i, url in enumerate(urls):
    Image.open(BytesIO(requests.get(url).content)).convert("RGB").save(
        save_path / f"{i}.jpg"
    )
# `instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `cat_toy`
instance_prompt = "<cat-toy> toy"  # {type:"string"}
# Generate Class Images
# Load the Stable Diffusion model
# Load models and create wrapper for stable diffusion
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)
# Setting up all training args
args = Namespace(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    resolution=vae.sample_size,
    center_crop=True,
    instance_data_dir=save_path,
    instance_prompt=instance_prompt,
    learning_rate=5e-06,
    max_train_steps=300,
    save_steps=50,
    train_batch_size=2,  # set to 1 if using prior preservation
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    mixed_precision="fp16",  # set to "fp16" for mixed-precision training.
    gradient_checkpointing=True,  # set this to True to lower the memory usage.
    use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
    seed=3434554,
    sample_batch_size=2,
    num_class_images=12,
    lr_scheduler="constant",
    lr_warmup_steps=100,
    output_dir=Path("dreambooth-concept"),
)
accelerate.notebook_launcher(
    train, args=(text_encoder, tokenizer, vae, unet, args), num_processes=1
)
for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
    if param.grad is not None:
        del param.grad  # free some memory
    torch.cuda.empty_cache()
