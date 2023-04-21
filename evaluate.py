#!/usr/bin/python
from argparse import ArgumentParser, Namespace
import clip
from pathlib import Path
from PIL import Image
import re
import torch
from torch.nn.functional import log_softmax


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--evaluation-directory",
        help="directory containing CLIP evaluation images whose filenames are their corresponding prompts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-x",
        "--exclude-words",
        default=[],
        help="words that, if found in an evaluation prompt, exclude that prompt from being used",
        nargs="*",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="directory containing DreamFusion output images whose filenames are their corresponding prompts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print cosine similarity for all prompts",
    )
    return parser.parse_args()


arguments = parse_arguments()
exclude_pattern = re.compile(
    fr"\b({'|'.join(arguments.exclude_words)})"
    if len(arguments.exclude_words) != 0
    else r"$^",
    re.IGNORECASE,
)
evaluation_paths = [
    evaluation_path
    for evaluation_path in arguments.evaluation_directory.iterdir()
    if not exclude_pattern.search(evaluation_path.stem)
]
output_paths = list(arguments.output_directory.iterdir())
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
device = clip_model.positional_embedding.device
evaluation_zs = torch.vstack(
    [
        clip_model.encode_text(clip.tokenize(evaluation_path.stem).to(device))
        for evaluation_path in evaluation_paths
    ]
)
cosine_similarity_true = 0.0
negative_log_likelihood = 0.0
accuracy = 0.0
for output_path in output_paths:
    output_text_z = clip_model.encode_text(clip.tokenize(output_path.stem).to(device))
    output_image_z = clip_model.encode_image(
        torch.unsqueeze(
            clip_preprocess(Image.open(output_path).convert("RGB")).to(device), 0
        )
    )
    cosine_similarity = torch.sum(
        torch.vstack([output_text_z, evaluation_zs]) * output_image_z, 1
    )
    argmax = torch.argmax(cosine_similarity)
    if arguments.verbose:
        print(f"{output_path}:")
        for index, (score, path) in enumerate(
            zip(cosine_similarity, [output_path, *evaluation_paths])
        ):
            print(f"{'*' if index == argmax else ' '} {score.item():.3f} {path.stem}")
    cosine_similarity_true += cosine_similarity[0].item()
    negative_log_likelihood -= log_softmax(cosine_similarity, 0)[0].item()
    accuracy += argmax == 0
output_count = len(output_paths)
print(f"mean cosine similarity of true prompt: {cosine_similarity_true / output_count}")
print(f"mean loss of true prompt: {negative_log_likelihood / output_count}")
print(f"mean accuracy: {accuracy / output_count}")
