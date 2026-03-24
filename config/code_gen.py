dataset = 'code-gen'
device = 'cuda'
compile = False

out_dir = 'out-code-gen'

eval_interval = 200
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'code-gen'
wandb_run_name = 'nanogpt'

n_layer = 6
n_head = 6
n_embd = 360

dropout = 0.1

learning_rate = 2e-4
min_lr = 2e-5

max_iters = 8000
lr_decay_iters = max_iters

batch_size = 32
gradient_accumulation_steps = 2
block_size = 128

weight_decay = 0.05

beta2 = 0.99

dtype = 'float16'

warmup_iters = 200

