dataset = 'rocstories'
device = 'cuda' # or 'cpu'
compile = False

out_dir = 'out-rocstories'
eval_interval = 200
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'nanogpt'

dataset = 'rocstories'

batch_size = 128 # 128
gradient_accumulation_steps = 1 # 2
block_size = 128  # important for stories

# model size (balanced for performance vs speed)
n_layer = 7
n_head = 6
n_embd = 384

dropout = 0.2
weight_decay = 0.06
# 6e-5
learning_rate = 2.5e-4   # VERY IMPORTANT (lower than shakespeare)
max_iters = 12000
lr_decay_iters = 12000
min_lr = 2e-5
# 1e-7
beta2 = 0.99

dtype = 'float16' # 'float16' or 'bfloat16' or 'float32'

warmup_iters = 200