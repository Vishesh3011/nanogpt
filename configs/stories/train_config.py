"""
Stories training config for Colab: scaled-up (~50M param) preset for
training on TinyStories with more compute.

Usage:
    python -m models.stories.train configs/stories/train_config_colab.py
"""

out_dir = "checkpoints/stories"
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "nanogpt-stories"
wandb_run_name = "stories-50M-colab"

batch_size = 64
gradient_accumulation_steps = 4
block_size = 256

# model size (~50M params)
n_layer = 8
n_head = 8
n_embd = 512

dropout = 0.1
weight_decay = 0.05

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5

beta1 = 0.9
beta2 = 0.95
dtype = "bfloat16"
warmup_iters = 500
compile = True
