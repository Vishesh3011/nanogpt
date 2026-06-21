"""
Code Colab training config — ~49M param scaled preset for CodeSearchNet Go.

Improvements over the 31M assignment config:
  - 12 layers, n_embd=448, block_size=512 (deeper + longer context)
  - GQA with n_kv_head=2 (saves params in attention, spent on depth)
  - RMSNorm + SwiGLU (LLaMA-style, best for code)
  - Larger effective batch: 32 × 4 = 128 sequences
  - bfloat16 + torch.compile

Usage:
    python -m models.code.train configs/code/train_config_colab.py
"""

# I/O
out_dir                = "checkpoints/code"
eval_interval          = 250
eval_iters             = 100
log_interval           = 10
always_save_checkpoint = True
init_from              = "scratch"

# wandb
wandb_log      = False
wandb_project  = "nanogpt-code"
wandb_run_name = "code-49M-codesearchnet"

# batching — effective batch = 32 × 4 = 128 sequences × 512 tokens
batch_size                  = 32
gradient_accumulation_steps = 4
block_size                  = 512

# model architecture (must match models/code/model.py scaled_model_args)
n_layer   = 12
n_head    = 8
n_kv_head = 2
n_embd    = 448
bias      = False
norm_type = "rmsnorm"
mlp_type  = "swiglu"
dropout   = 0.1

# optimiser
learning_rate = 3e-4
max_iters     = 15000
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# LR schedule
decay_lr       = True
warmup_iters   = 300
lr_decay_iters = 15000
min_lr         = 3e-5

# system
device  = "cuda"
dtype   = "bfloat16"
compile = True
seed    = 1337
