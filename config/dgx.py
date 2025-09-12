import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/model-pretrained/flow_grpo/AI-ModelScope/stable-diffusion-3___5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/model-pretrained/flow_grpo/AI-ModelScope/stable-diffusion-3___5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    config.resolution = 512
    config.sample.train_batch_size = 12
    config.sample.num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 12
    config.sample.test_batch_size = 16 # 11 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.001
    # kl reward
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/model-pretrained/flow_grpo/AI-ModelScope/stable-diffusion-3___5-medium"
    config.sample.num_steps = 10 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Flow-RLG/model-pretrained/flow_grpo/AI-ModelScope/stable-diffusion-3___5-medium"
    config.sample.num_steps = 5 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 4*A100
    config.resolution = 512
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 6
    config.sample.test_batch_size = 16 # The test set has a total of 2048

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
