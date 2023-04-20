#!/usr/bin/python
import accelerate
from argparse import ArgumentParser, Namespace
from diffusers import AutoencoderKL, UNet2DConditionModel
import itertools
from pathlib import Path
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from train_dreambooth import train


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "-p",
        "--instance_prompt",
        default="<cat-toy> toy",
        help="a prompt that should contain a good description of what your object or style is, together with the initializer word",
        type=str,
    )
    return parser.parse_args()


# `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
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
arguments = parse_arguments()
arguments = Namespace(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    resolution=vae.sample_size,
    center_crop=True,
    instance_data_dir=arguments.directory / "images",
    instance_prompt=arguments.instance_prompt,
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
    output_dir=arguments.directory / "dreambooth",
)
accelerate.notebook_launcher(
    train, args=(text_encoder, tokenizer, vae, unet, arguments), num_processes=1
)
for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
    if param.grad is not None:
        del param.grad  # free some memory
    torch.cuda.empty_cache()
