"""
Stories Colab training config — ~49M param scaled preset for TinyStories.

Improvements over the 31M assignment config:
  - 9 layers, n_embd=480, block_size=512 (more capacity + longer context)
  - RoPE (no wpe), full MHA, LayerNorm+GELU (GPT-2 style but modernised)
  - Larger effective batch: batch_size=32 x grad_accum=8 = 256 sequences
  - bfloat16 + torch.compile for Colab A100 / T4
  - Cosine LR from 3e-4 → 3e-5 over 20k iters

Usage:
    python -m models.stories.train configs/stories/train_config_colab.py
"""

# I/O
out_dir               = "checkpoints/stories"
eval_interval         = 250
eval_iters            = 100
log_interval          = 10
always_save_checkpoint = True
init_from             = "scratch"

# wandb (enable to track Colab runs)
wandb_log      = False
wandb_project  = "nanogpt-stories"
wandb_run_name = "stories-49M-tinystories"

# batching — effective batch = 32 × 8 = 256 sequences × 512 tokens
batch_size                 = 32
gradient_accumulation_steps = 8
block_size                 = 512

# model architecture (must match models/stories/model.py scaled_model_args)
n_layer   = 9
n_head    = 8
n_kv_head = 8
n_embd    = 480
bias      = True
norm_type = "layernorm"
mlp_type  = "gelu"
dropout   = 0.1

# optimiser
learning_rate = 3e-4
max_iters     = 20000
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# LR schedule
decay_lr       = True
warmup_iters   = 500
lr_decay_iters = 20000
min_lr         = 3e-5

# system
device  = "cuda"
dtype   = "bfloat16"
compile = True
seed    = 1337
