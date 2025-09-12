CUDA_VISIBLE_DEVICES=0 /inspire/hdd/project/embodied-multimodality/public/lzjjin/anaconda3/envs/rlg/bin/python /inspire/hdd/project/embodied-multimodality/public/lzjjin/Reinforcement-learning-guidance/scripts/generate.py \
  --config="config/dgx.py:pickscore_sd3" \
  --lora_path="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/model-pretrained/flow_grpo/models--jieliu--SD3.5M-FlowGRPO-PickScore/snapshots/10c56697459bbdbe54d5e375912f49a0bcfae773" \
  --tuned_guidance_scale=2.4 \
  --output_dir="logs/aes_generated_images/" \
  --prompt_file="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Reinforcement-learning-guidance/dataset/pickscore/test.txt"
