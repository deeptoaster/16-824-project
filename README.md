# Few-Shot 3D with DreamBooth and DreamFusion

## Instructions

### Setup

1.  Clone this repo recursively (`--recurse-submodules`).
2.  Install [CLIP](https://github.com/openai/CLIP) with `pip`. This should pull
    in all additional requirements.
3.  Install _stable-dreamfusion/requirements.txt_ with `pip`.
4.  Change line 54 of _stable-dreamfusion/main.py_ to remove the _choices_
    argument.
5.  Change line 61 of _stable-dreamfusion/sd.py_ from an exception to
    `model_key = self.sd_version`.

This can be accomplished in the CLI as follows:

```
$ git clone --recurse-submodules https://github.com/deeptoaster/16-824-project
$ cd 16-824-project
$ python -m venv env
$ source env/bin/activate
(env) $ pip install --requirement requirements.txt
(env) $ pip install --requirement stable-dreamfusion/requirements.txt
(env) $ sed -i 's/^\(\s*parser\.add_argument('\''--sd_version'\''.*\), choices=\[.*\]\(.*\)$/\1\2/' stable-dreamfusion/main.py
(env) $ sed -i 's/^\(\s*\)raise ValueError(.*{self\.sd_version\}.*$/\1model_key = self.sd_version/' stable-dreamfusion/sd.py
```

### Training

1.  Place three to five images of the custom concept in a dedicated directory.
2.  Run DreamBooth with _run_dreambooth.py_. `--instance_prompt` should contain
    a unique tag followed by a description, such as `"<cat-toy> toy"`. Pass the
    parent of the directory containing the images.
3.  Train DreamFusion with _stable-dreamfusion/main.py_ according to the [usage
    documentation](https://github.com/ashawkey/stable-dreamfusion#usage),
    passing in the output folder of the previous step (a sister folder to the
    one containing the images, named _dreambooth_) as the parameter
    `--sd_version`.
4.  Render DreamFusion with the same script and parameters.

This can be accomplished in the CLI as follows (using the same venv as before):

```
(env) $ mkdir --parents exp/cat-toy/images
(env) $ cd exp/cat-toy/images
(env) $ wget https://huggingface.co/datasets/valhalla/images/resolve/main/{2,3,5,6}.jpeg
(env) $ cd ../../..
(env) $ python run_dreambooth.py --instance_prompt="<cat-toy> toy" exp/cat-toy
(env) $ python stable-dreamfusion/main.py --text "a <cat-toy> toy" --workspace exp/cat-toy/dreamfusion -O2 --sd_version exp/cat-toy/dreambooth
(env) $ python stable-dreamfusion/main.py --workspace exp/cat-toy/dreamfusion -O2 --sd_version exp/cat-toy/dreambooth --test
```

### Evaluation

1.  Download and extract the [COCO evaluation
    images](https://openaipublic.azureedge.net/main/point-e/coco_images.zip).
2.  Extract desired stills from the videos rendered by DreamFusion into a
    dedicated directory. The filename of each image should be the prompt used
    to generate it followed by the image extension; see the COCO evaluation
    images for examples.
3.  Run CLIP R-Precision with _evaluate.py_, using either of the two COCO image
    sets and optionally excluding evaluation images whose prompts contain
    certain words. For example, we might generate using the prompt "there is a
    <cat-toy> toy on the table," then exclude _there is a cell phone on a
    table..png_.

This can be accomplished in the CLI as follows (using the same venv as before):

```
(env) $ wget https://openaipublic.azureedge.net/main/point-e/coco_images.zip
(env) $ unzip coco_images.zip
(env) $ mkdir --parents exp/cat-toy/output
(env) $ ffmpeg -i exp/cat-toy/dreamfusion/results/df_ep0100_rgb.mp4 -frames:v 1 exp/cat-toy/output/a\ \<cat-toy\>\ toy.png
(env) $ python evaluate.py -e coco_images/1 -x cat toy -o exp/cat-toy/output
```
