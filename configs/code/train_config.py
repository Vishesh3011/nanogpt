"""
Code-generation training config: ~31M-param preset from the original
assignment (formerly config/code_gen.py).

Usage:
    python -m models.code.train configs/code/train_config.py
"""

out_dir = "checkpoints/code"
eval_interval = 200
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "nanogpt-code"
wandb_run_name = "code-31M"

n_layer = 6
n_head = 7
n_embd = 350

dropout = 0.1
learning_rate = 2e-4
min_lr = 2e-5
max_iters = 8000
lr_decay_iters = 8000

batch_size = 64
gradient_accumulation_steps = 2
block_size = 128

weight_decay = 0.05
beta2 = 0.99
dtype = "float16"
warmup_iters = 200
