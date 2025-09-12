import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm

def aesthetic_score():
    from rlg.aesthetic_scorer import AestheticScorer
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
    from rlg.pickscore_scorer import PickScoreScorer
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
    from rlg.imagereward_scorer import ImageRewardScorer
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

def evaluate_images_in_directory(directory_path, scoring_fn, batch_size=1):
    """
    评估单个文件夹中的所有图像，并为每张图片返回一个详细的分数记录。
    """
    print(f"\n--- 正在处理目录: {directory_path} ---")
    
    metadata_path = os.path.join(directory_path, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"错误: 在 {directory_path} 中找不到 metadata.csv。")
        return None

    metadata_rows = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_rows.append(row)

    if not metadata_rows:
        print(f"警告: {directory_path} 中的 metadata.csv 为空。正在跳过。")
        return None

    transform = transforms.Compose([transforms.ToTensor()])
    all_results = []
    
    for i in tqdm(range(0, len(metadata_rows), batch_size), desc="正在评估批次"):
        batch_data = metadata_rows[i:i+batch_size]
        image_paths_batch = [os.path.join(directory_path, row['image_path']) for row in batch_data]
        prompts_batch = [row['prompt'] for row in batch_data]

        try:
            images_tensors = [transform(Image.open(p).convert('RGB')) for p in image_paths_batch]
            images_batch_tensor = torch.stack(images_tensors)
        except FileNotFoundError as e:
            print(f"错误: 找不到图片 - {e}。正在跳过此批次。")
            continue
        
        scores_dict, _ = scoring_fn(images_batch_tensor, prompts_batch, metadata={'paths': image_paths_batch})
        
        for j in range(len(batch_data)):
            result_row = {
                'image_path': batch_data[j]['image_path'],
                'prompt': batch_data[j]['prompt']
            }
            for score_name, score_values in scores_dict.items():
                score = score_values[j]
                result_row[f'{score_name}_score'] = score.item() if hasattr(score, 'item') else score
            
            all_results.append(result_row)

    if not all_results:
        print("警告: 没有计算出任何分数。")
        return None

    print(f"--- 目录 {os.path.basename(directory_path)} 处理完成，共评估 {len(all_results)} 张图片 ---")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="为目录中的每张图片进行评估，并将详细分数保存到CSV文件中。")
    parser.add_argument("--input_dir", type=str, default='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/logs/aes_generated_images/scale_0.0', help="包含图片和 'metadata.csv' 的输入目录。")
    parser.add_argument("--batch_size", type=int, default=1, help="评估时的批处理大小。")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    score_dict = {
        "aesthetic": 1.0,
        "imagereward": 1.0,
        "pickscore": 1.0,
    }
    
    scoring_fn = multi_score(device, score_dict)
    
    output_csv_path = os.path.join(args.input_dir, 'evaluation_results.csv')

    all_image_results = evaluate_images_in_directory(args.input_dir, scoring_fn, args.batch_size)
    
    if not all_image_results:
        print("\n没有评估结果可供保存。程序退出。")
        return

    print(f"\n正在将 {len(all_image_results)} 条详细结果保存到 {output_csv_path}...")
    
    fieldnames = list(all_image_results[0].keys())

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_image_results)
        print(f"✅ 详细评估结果已成功保存到 {output_csv_path}")
    except IOError as e:
        print(f"错误: 无法写入文件 {output_csv_path}。错误信息: {e}")

    # ✅ 新增: 计算并打印平均分
    print("\n--- 平均分统计 ---")
    score_keys = [key for key in fieldnames if key.endswith('_score')]
    num_images = len(all_image_results)

    for key in score_keys:
        total_score = sum(result[key] for result in all_image_results)
        mean_score = total_score / num_images
        # 清理名称用于打印 (例如 'aesthetic_score' -> 'Aesthetic')
        clean_name = key.replace('_score', '').capitalize()
        print(f"平均 {clean_name} 分数: {mean_score:.4f}")

    print("\n✅ 完成。")


if __name__ == "__main__":
    main()
