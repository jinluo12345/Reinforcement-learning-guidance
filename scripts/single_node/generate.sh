CUDA_VISIBLE_DEVICES=2 /inspire/hdd/project/embodied-multimodality/public/lzjjin/anaconda3/envs/flow_grpo/bin/python /inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/scripts/generate.py \
  --config="config/dgx.py:pickscore_sd3" \
  --lora_path="model-pretrained/flow_grpo/models--jieliu--SD3.5M-FlowGRPO-PickScore/snapshots/10c56697459bbdbe54d5e375912f49a0bcfae773" \
  --tuned_guidance_scale=2.4 \
  --output_dir="logs-exp-new/aes_generated_images_reward_scale_1.5_0.5_fix_reward/" \
  --prompt_file="Flow-RLG/dataset/pickscore/test_small.txt"
