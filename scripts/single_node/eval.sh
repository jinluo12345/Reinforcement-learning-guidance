#!/bin/bash
module purge
module load miniforge3/24.1
source /home/bingxing2/apps/package/pytorch/2.6.0-cu124-cp311/env.sh
source activate flow_grpo

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29503 scripts/evaluate.py --config config/dgx.py:pickscore_sd3

# salloc --gres=gpu:1 -p vip_gpu_ailab -A aim
# salloc: Granted job allocation 823749
# salloc: Waiting for resource configuration
# salloc: Nodes paraai-n32-h-01-agent-201 are ready for job

# sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A aim scripts/single_node/eval.sh
# 823779 1.0, 823781 1.2

# sbatch -N 1 --gres=gpu:1 -p vip_gpu_ailab_low -A ailab evaluate.sh