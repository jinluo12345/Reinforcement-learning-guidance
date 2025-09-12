<h1 align="center"> RLG:<br>Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance </h1>
<!-- <div align="center">
  <a href='https://arxiv.org/abs/2505.05470'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://gongyeliu.github.io/Flow-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp;
  <a href="https://github.com/yifan123/flow_grpo"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp;
</div> -->

## Overview
This repository is an implementation of Reinforcement Learning Guidance (RLG), a novel inference-time method for enhancing and controlling the alignment of diffusion models. RLG adapts Classifier-Free Guidance (CFG) by combining the outputs of a base model and an RL-fine-tuned model using a geometric average, enabling dynamic control over alignment strength without additional training. Our work is built upon the [Flow-GRPO repository](https://github.com/yifan123/flow_grpo), extending its capabilities to support flexible alignment control for various downstream tasks, such as human preference alignment, compositional generation, text rendering.


<!-- ## 🤗 Model
| Task    | Model |
| -------- | -------- |
| GenEval     | [🤗GenEval](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-GenEval) |
| Text Rendering     | [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text) |
| Human Preference Alignment     | [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore) | -->


## 🚀 Quick Started
### 1. Environment Set Up
Clone this repository and install packages.
```bash
conda create -n rlg python=3.10.16
pip install -e .
```
### 2. Model Downloads

Our reinfoecement-learning-guidance approach requires both the reference model and the RL-finetuned model to generate images, so please download them in advance.

#### Base Model

* **SD3.5**: Access at `stabilityai/stable-diffusion-3.5-medium`.

#### Flow-GRPO Models

* **GenEval**: Available at [🤗GenEval](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-GenEval).
* **Text Rendering**: Available at [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text).
* **Human Preference Alignment**: Available at [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore).

#### Eval Models

* **PickScore**:
  * `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  * `yuvalkirstain/PickScore_v1`
* **Aesthetic Score**: `openai/clip-vit-large-patch14`
* **ImageReward Score**: `zai-org/ImageReward`


### 3. Image Generation
#### Running Inference with RLG

To generate images using the Reinforcement Learning Guidance (RLG) framework with Flow-GRPO, use the following command:
```python
python /inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/scripts/generate.py \
  --config="config/dgx.py:pickscore_sd3" \
  --lora_path="model-pretrained/flow_grpo/models--jieliu--SD3.5M-FlowGRPO-PickScore/snapshots/10c56697459bbdbe54d5e375912f49a0bcfae773" \
  --tuned_guidance_scale=2.4 \
  --output_dir="logs/aes_generated_images/" \
  --prompt_file="Flow-RLG/dataset/pickscore/test.txt"

```
### Parameter Descriptions

* \--config: Defines the configuration file and specific setup for the model. The value "config/dgx.py\:pickscore\_sd3" refers to the pickscore\_sd3 configuration in the dgx.py file
* \--lora\_path: Specifies the path to the LoRA (Low-Rank Adaptation) weights for the RL-fine-tuned model. 
* \--tuned\_guidance\_scale: Adjusts the strength of RL-guided alignment during inference. Recommended range: 1.0 to 3.0, where higher values enhance alignment but may reduce diversity.
* \--output\_dir: Directory where generated images are saved. For example, "logs/aes\_generated\_images/" specifies the output folder for storing results.
* \--prompt\_file: Path to the text file containing prompts for image generation. 

---

### 4. Eval model Preparation
The steps above only install the tools used to generate. Since each evaluation model may rely on different versions, combining them in one Conda environment can cause version conflicts. To avoid this, you only need to install the specific reward model you plan to use.
<!-- 
#### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server). -->

#### OCR
Please install paddle-ocr:
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```
Then, pre-download the model using the Python command line:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

#### Pickscore
PickScore requires no additional installation.

#### ImageReward
Please install imagereward:
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```



## 🤗 Acknowledgement
This repo is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo). We thank the authors for their valuable contributions to the AIGC community. Special thanks to Kevin Black for the excellent *ddpo-pytorch* repo.

<!-- ## ⭐Citation
If you find Flow-GRPO useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```
@article{liu2025flow,
  title={Flow-grpo: Training flow matching models via online rl},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```
If you find Flow-DPO useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```
@article{liu2025improving,
  title={Improving video generation with human feedback},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Yuan, Ziyang and Liu, Xiaokun and Zheng, Mingwu and Wu, Xiele and Wang, Qiulin and Qin, Wenyu and Xia, Menghan and others},
  journal={arXiv preprint arXiv:2501.13918},
  year={2025}
}
``` -->