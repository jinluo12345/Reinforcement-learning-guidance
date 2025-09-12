from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import ImageReward as RM

class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.model_path = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/DiffusionDPO/hf_models/ImageReward/ImageReward.pt"
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device, med_config='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/DiffusionDPO/hf_models/ImageReward/med_config.json').eval().to(dtype=dtype)
        self.model.requires_grad_(False)
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        _, rewards = self.model.inference_rank(prompt, images)
        return rewards

# Usage example
def main():
    scorer = ImageRewardScorer(
        device="cuda",
        dtype=torch.float32
    )

    images=[
    "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/logs/aes_generated_images/scale_1.2/02047.png",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        '8 years old , handsome blue-skinned Hindu God Krishna, realistic black hair, detailed texture, pretty,cute sharp bright big black eyes and pupils intricate, small nose, , elegant, realistic 3D render, epic,detailed digital painting, artstation, concept art, matte, GLOBAL ILLUMINATION sharp focus, illustration, art by artgerm and alphonse mucha'
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()