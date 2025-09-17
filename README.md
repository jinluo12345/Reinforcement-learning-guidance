<h1 align="center"> RLG:<br>Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2508.21016'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <!-- <a href='https://gongyeliu.github.io/Flow-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp; -->
  <a href="https://github.com/jinluo12345/Reinforcement-learning-guidance"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <!-- <a href='https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp; -->
</div>

## ✨ Overview

This repository presents **Reinforcement Learning Guidance (RLG)**, an innovative inference-time method designed to enhance and control the alignment of diffusion models. RLG builds upon the widely used Classifier-Free Guidance (CFG) by introducing a novel approach: it harmonizes the outputs of a base diffusion model and an RL-fine-tuned model through a geometric average. This unique combination empowers users with dynamic and precise control over alignment strength *without requiring any additional training*.


<!-- ## 🤗 Model
| Task    | Model |
| -------- | -------- |
| GenEval     | [🤗GenEval](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-GenEval) |
| Text Rendering     | [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text) |
| Human Preference Alignment     | [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore) | -->


## 🚀 Quick Start Guide

Let's get RLG up and running! Follow these simple steps to set up your environment and generate stunning images.

### 1. 🌐 Environment Setup

Begin by cloning the repository and installing the necessary packages.

```bash
git clone https://github.com/jinluo12345/Reinforcement-learning-guidance.git
cd Reinforcement-learning-guidance
conda create -n rlg python=3.10.16 -y
conda activate rlg # Activate your new environment
pip install -e .
```

### 2. ⬇️ Model Downloads

RLG requires both a reference base model and an RL-fine-tuned model to operate. Please download them in advance.

#### Base Model

RLG currently primarily supports `stable-diffusion-3.5` as its base model.

* **SD3.5**: Accessible via `stabilityai/stable-diffusion-3.5-medium`.

#### RL-fine-tuned Model

RLG currently supports models fine-tuned with Flow-GRPO.

* **PickScore Alignment**: Download from [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore).
* **Text Rendering**: Download from [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text).

#### Evaluation Models

For comprehensive evaluation, you might need these models:

* **PickScore**:

  * `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  * `yuvalkirstain/PickScore_v1`
* **Aesthetic Score**: `openai/clip-vit-large-patch14`
* **ImageReward Score**: `zai-org/ImageReward`

### 3. 🖼️ Image Generation

Unleash the power of RLG to generate images!

#### Running Inference with RLG

To generate images using the Reinforcement Learning Guidance (RLG) framework, use the following command:

```python
python scripts/generate.py \
  --config="config/dgx.py:pickscore_sd3" \
  --lora_path="{your downloaded RL-finetuned model's ckpt path}" \
  --tuned_guidance_scale=1.6 \
  --output_dir="logs/" \
  --prompt_file="dataset/pickscore/test.txt"
```

**Note:** If you wish to generate images using the *original base model* or the *RL-fine-tuned model without RLG*, simply set `tuned_guidance_scale` accordingly:

* Set `tuned_guidance_scale` to `0.0` for the original base model.
* Set `tuned_guidance_scale` to `1.0` for the original RL-fine-tuned model (sampling without RLG).

#### Parameter Descriptions

* `--config`: Specifies the configuration file and model setup (e.g., `config/dgx.py:pickscore_sd3` refers to the `pickscore_sd3` configuration in `dgx.py`, fine-tuned on PickScore reward).
* `--lora_path`: The file path to the LoRA (Low-Rank Adaptation) weights of your downloaded RL-fine-tuned model.
* `--tuned_guidance_scale`: Adjusts the strength of RL-guided alignment during inference.

  * **Recommended range**: `1.0` to `3.0`.
  * Higher values enhance alignment but might reduce image diversity.
* `--output_dir`: The directory where all your magnificent generated images will be saved.
* `--prompt_file`: Path to a text file containing the prompts for image generation.

---

### 4. 📈 Evaluation Model Preparation

The above steps install tools for generation. Evaluation models, however, may have specific dependency versions that could conflict if installed in a single environment. To prevent this, install only the specific reward models you intend to use.

#### PickScore

PickScore requires no additional installation steps. It's ready to go!

#### ImageReward

To use ImageReward, install it along with the CLIP library:

```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

#### Running Evaluations

We provide scripts to evaluate Aesthetic Score, PickScore, and ImageReward. To run evaluations, use the following command:

```bash
python scripts/cal_aes.py --input_dir logs/ --batch_size 1
```

This script will by default evaluate all supported rewards and save the results into a CSV file within your input directories.

**Customizing Evaluation:** If you wish to evaluate only specific scores (e.g., just PickScore), you can modify the `score_dict` within `cal_aes.py` by setting other scores to `0.0`:

```python
# Example to evaluate only PickScore
score_dict = {
    "aesthetic": 0.0,
    "imagereward": 0.0,
    "pickscore": 1.0, # Enable PickScore evaluation
}
```

## 🤗 Acknowledgement
This repo is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo). We thank the authors for their valuable contributions to the AIGC community. Special thanks to Liu Jie for the excellent *flow_grpo* repo.

<!-- ## ⭐Citation
If you find RLG useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```
@misc{jin2025inferencetimealignmentcontroldiffusion,
      title={Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance}, 
      author={Luozhijie Jin and Zijie Qiu and Jie Liu and Zijie Diao and Lifeng Qiao and Ning Ding and Alex Lamb and Xipeng Qiu},
      year={2025},
      eprint={2508.21016},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.21016}, 
}
``` -->
