import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm

# --- Environment and Cache Configuration ---
CACHE_DIR = "model-pretrained"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TORCH_HOME'] = CACHE_DIR

# ====================================================================================
#  Scorer Definitions
# ====================================================================================
def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer
    scorer = AestheticScorer(dtype=torch.float32).cuda()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}
    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer
    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}
    return _fn

def imagereward_score(device):
    """
    Loads the original ImageRewardScorer but modifies the input logic.
    """
    from flow_grpo.imagereward_scorer import ImageRewardScorer
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        image_paths = metadata.get('paths')
        if not image_paths:
            raise ValueError("ImageReward scorer requires 'paths' in metadata but none were found.")

        pil_images = [Image.open(p).convert('RGB') for p in image_paths]
        scores = scorer(prompts, pil_images)
        if not isinstance(scores,list):
            scores=[scores]
        return scores, {}
        
    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        if 'device' in score_functions[score_name].__code__.co_varnames:
            score_fns[score_name] = score_functions[score_name](device)
        else:
            score_fns[score_name] = score_functions[score_name]()
    def _fn(images, prompts, metadata):
        score_details = {}
        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
        return score_details, {}
    return _fn

# ====================================================================================
#  Main Evaluation Logic
# ====================================================================================

def evaluate_directory(directory_path, scoring_fn, batch_size=16):
    """
    Evaluates images in a single folder and returns the mean of all scores.
    """
    print(f"\n--- Processing directory: {directory_path} ---")
    
    metadata_path = os.path.join(directory_path, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.csv not found in {directory_path}. Skipping.")
        return None

    metadata_rows = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_rows.append(row)

    if not metadata_rows:
        print(f"Warning: metadata.csv in {directory_path} is empty. Skipping.")
        return None

    transform = transforms.Compose([transforms.ToTensor()])
    all_scores = defaultdict(list)
    
    for i in tqdm(range(0, len(metadata_rows), batch_size), desc="Evaluating batches"):
        batch_data = metadata_rows[i:i+batch_size]
        image_paths_batch = [os.path.join(directory_path, row['image_path']) for row in batch_data]
        prompts_batch = [row['prompt'] for row in batch_data]

        try:
            images_tensors = [transform(Image.open(p).convert('RGB')) for p in image_paths_batch]
            images_batch_tensor = torch.stack(images_tensors)
        except FileNotFoundError as e:
            print(f"Error: Image not found - {e}. Skipping batch.")
            continue
        
        scores, _ = scoring_fn(images_batch_tensor, prompts_batch, metadata={'paths': image_paths_batch})
        
        for score_name, score_values in scores.items():
            all_scores[score_name].extend([s.item() if hasattr(s, 'item') else s for s in score_values])

    if not all_scores:
        print("Warning: No scores were calculated. Skipping.")
        return None

    mean_scores = {f"mean_{name}": np.mean(values) for name, values in all_scores.items()}
    
    print(f"--- Directory Summary: {os.path.basename(directory_path)} ---")
    for name, mean_val in mean_scores.items():
        print(f"{name}: {mean_val:.4f}")
    
    return mean_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images and save mean scores.")
    parser.add_argument("--base_dir", type=str, default='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/logs-exp-new/aes_generated_images_reward_scale_1.5_0.5_fix_reward', help="The base directory containing scale_* subfolders.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    score_dict = {
        "aesthetic": 1.0,
        "imagereward": 1.0,
        "pickscore": 1.0,
    }
    
    scoring_fn = multi_score(device, score_dict)
    
    output_csv_path = os.path.join(args.base_dir, 'mean_evaluation_summary.csv')
    processed_directories = set()

    # ✅ MODIFICATION: Read existing CSV to find already processed directories
    file_exists = os.path.exists(output_csv_path)
    if file_exists:
        try:
            with open(output_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'directory' in row:
                        processed_directories.add(row['directory'])
            print(f"Found {len(processed_directories)} previously evaluated directories in {output_csv_path}.")
        except (IOError, csv.Error) as e:
            print(f"Warning: Could not read existing summary file. Will start fresh. Error: {e}")
            processed_directories.clear()


    newly_evaluated_results = []

    # Sort entries to ensure consistent order
    for entry in sorted(os.scandir(args.base_dir), key=lambda e: e.name):
        if entry.is_dir() and entry.name.startswith("scale_"):
            
            # ✅ MODIFICATION: Skip if directory is already in the CSV
            if entry.name in processed_directories:
                print(f"Skipping '{entry.name}' as it's already in the summary CSV.")
                continue

            mean_scores = evaluate_directory(entry.path, scoring_fn, args.batch_size)
            
            if mean_scores:
                result_row = {'directory': entry.name}
                result_row.update(mean_scores)
                newly_evaluated_results.append(result_row)

    if not newly_evaluated_results:
        print("\nNo new directories to evaluate or save. Exiting.")
        return

    print(f"\nSaving {len(newly_evaluated_results)} new results to {output_csv_path}...")
    
    # Dynamically create fieldnames from the first new result
    fieldnames = ['directory'] + list(newly_evaluated_results[0].keys())[1:]

    # ✅ MODIFICATION: Open in append mode ('a')
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if the file is new
        if not file_exists or os.path.getsize(output_csv_path) == 0:
            writer.writeheader()
            
        writer.writerows(newly_evaluated_results)
        
    print("✅ Done. Summary CSV updated successfully.")

if __name__ == "__main__":
    main()
