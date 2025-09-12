import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

# Import the necessary functions from your project
from rlg.diffusers_patch.sd3_pipeline_with_rlg import pipeline_with_logprob
from rlg.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

# Setup logging and progress bars
tqdm = partial(tqdm, dynamic_ncols=True)
from accelerate.logging import get_logger
logger = get_logger(__name__)

# --- Define Command-Line Flags ---
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("lora_path", None, "Path to the LoRA checkpoint to load.")
flags.DEFINE_float("tuned_guidance_scale", 1.0, "The 'tuned' guidance scale for the pipeline.")
flags.DEFINE_string("output_dir", "./outputs", "The base directory to save generated images and metadata.")
flags.DEFINE_string("prompt_file", None, "Path to a text file containing prompts (one per line).")

# Mark required flags
flags.mark_flag_as_required("lora_path")
flags.mark_flag_as_required("prompt_file")


class TextPromptDataset(Dataset):
    """A simple dataset to load prompts from a text file."""
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        return prompts,


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    """Computes text embeddings for the given prompts."""
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def main(_):
    config = FLAGS.config
    # --- Setup Accelerator ---
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    logger.info(f"Accelerator configured for {accelerator.device} with mixed precision: {config.mixed_precision}")
    set_seed(config.seed, device_specific=True)
    # --- Prepare Directories ---
    # Create a specific subdirectory for this run based on the tuned guidance scale
    run_output_dir = os.path.join(FLAGS.output_dir, f"scale_{FLAGS.tuned_guidance_scale}")
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Outputs will be saved to: {run_output_dir}")

    # --- Load Models ---
    logger.info(f"Loading base model from: {config.pretrained.model}")
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    
    logger.info(f"Loading LoRA weights from: {FLAGS.lora_path}")
    pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, FLAGS.lora_path)
    logger.info("Successfully loaded LoRA weights onto the transformer.")

    # --- Configure Pipeline ---
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None

    pipeline.to(accelerator.device)
    pipeline.transformer.eval()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    
    # --- Prepare Dataset ---
    logger.info(f"Loading prompts from: {FLAGS.prompt_file}")
    prompt_dataset = TextPromptDataset(FLAGS.prompt_file)
    dataloader = DataLoader(
        prompt_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
    )
    logger.info(f"Found {len(prompt_dataset)} prompts to generate.")

    # --- Prepare for Generation ---
    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)
    autocast = accelerator.autocast

    # Pre-compute negative prompt embeddings (empty string)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=256, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)
    
    # --- Generation Loop ---
    metadata_records = []
    global_image_index = 0
    
    # Only show progress bar on the main process
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc="Generating Images")

    for batch in dataloader:
        prompts = batch[0]
        
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=256,
            device=accelerator.device
        )
        
        # Adjust negative prompts for the last batch if its size is smaller
        current_batch_size = len(prompts)
        if current_batch_size < config.sample.test_batch_size:
            batch_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            batch_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            batch_neg_prompt_embeds = sample_neg_prompt_embeds
            batch_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with autocast():
            with torch.no_grad():
                # Use the custom pipeline function with the tuned_guidance_scale flag
                images, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=batch_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=batch_neg_pooled_prompt_embeds,
                    num_inference_steps=20,
                    zero_init_step=0,
                    guidance_scale=config.sample.guidance_scale, # This is the standard CFG scale
                    tuned_guidance_scale=FLAGS.tuned_guidance_scale, # This is your custom scale
                    output_type="pt",
                    return_dict=False,
                    height=config.resolution,
                    width=config.resolution,
                    determistic=True,
                )

        # Process and save images and metadata
        for i, img_tensor in enumerate(images):
            prompt_text = prompts[i]
            
            # Format filename with zero-padding
            image_filename = f"{global_image_index:05d}.png"
            image_save_path = os.path.join(run_output_dir, image_filename)
            
            # Convert tensor to PIL Image and save
            img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_tensor * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            pil_image.save(image_save_path)
            
            # Store metadata for the CSV file
            absolute_image_path = os.path.abspath(image_save_path)
            metadata_records.append({"image_path": absolute_image_path, "prompt": prompt_text})
            
            global_image_index += 1
        
        pbar.update(1)
    
    pbar.close()

    # --- Save Metadata CSV ---
    if accelerator.is_main_process:
        csv_path = os.path.join(run_output_dir, "metadata.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["image_path", "prompt"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_records)
            logger.info(f"Successfully saved metadata to {csv_path}")
        except IOError as e:
            logger.error(f"Failed to write CSV file: {e}")

    logger.info("Image generation complete.")


if __name__ == "__main__":
    app.run(main)
