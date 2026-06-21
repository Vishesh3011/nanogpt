"""
Stories training config: the best-performing 31M-param preset from the
original assignment (formerly config/31M_best_25_21_9600.py).

Usage:
    python -m models.stories.train configs/stories/train_config.py
"""

out_dir = "checkpoints/stories"
eval_interval = 200
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "nanogpt-stories"
wandb_run_name = "stories-31M"

batch_size = 128
gradient_accumulation_steps = 1
block_size = 128

# model size (~31M params)
n_layer = 7
n_head = 6
n_embd = 384

dropout = 0.2
weight_decay = 0.06

learning_rate = 2.5e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 2e-5

beta2 = 0.99
dtype = "float16"
warmup_iters = 200
