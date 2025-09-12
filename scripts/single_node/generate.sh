python scripts/generate.py \
  --config="config/dgx.py:pickscore_sd3" \
  --lora_path="{your downloaded RL-finetuned model's ckpt path}" \
  --tuned_guidance_scale=1.6 \
  --output_dir="logs/" \
  --prompt_file="dataset/pickscore/test.txt"