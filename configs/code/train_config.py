"""
Code-generation training config for Colab: scaled-up (~50M param) preset.

Usage:
    python -m models.code.train configs/code/train_config_colab.py
"""

out_dir = "checkpoints/code"
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "nanogpt-code"
wandb_run_name = "code-50M-colab"

n_layer = 8
n_head = 8
n_embd = 512

dropout = 0.1
learning_rate = 3e-4
min_lr = 3e-5
max_iters = 15000
lr_decay_iters = 15000

batch_size = 64
gradient_accumulation_steps = 2
block_size = 256

weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
dtype = "bfloat16"
warmup_iters = 300
compile = True
