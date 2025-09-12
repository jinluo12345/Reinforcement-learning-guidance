import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 100
    config.save_freq = 20
    config.eval_freq = 100
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True
    config.dataset = ""
    config.resolution = 768

    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    pretrained.revision = "main"

    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 40
    sample.eval_num_steps = 40
    sample.eta = 1.0
    sample.guidance_scale = 4.5
    sample.train_batch_size = 1
    sample.num_image_per_prompt = 1
    sample.test_batch_size = 1
    sample.num_batches_per_epoch = 2
    sample.kl_reward = 0
    sample.global_std = False

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0
    train.beta = 0.0
    train.lora_path = None
    train.ema = False

    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    config.reward_fn = ml_collections.ConfigDict()
    config.save_dir = ''
    config.per_prompt_stat_tracking = True

    return config
