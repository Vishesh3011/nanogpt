dataset = 'rocstories'
device = 'cuda' # or 'cpu'
compile = False

out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'nanogpt'

dataset = 'rocstories'

batch_size = 64 # 128
gradient_accumulation_steps = 2 # 2
block_size = 256  # important for stories

# model size (balanced for performance vs speed)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15

learning_rate = 3e-4   # VERY IMPORTANT (lower than shakespeare)
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-6

beta2 = 0.99

warmup_iters = 100