dataset = 'rocstories'
device = 'cuda'
compile = False

out_dir = 'out-rocstories'
eval_interval = 200
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'nanogpt'

batch_size = 64
gradient_accumulation_steps = 2 # 2
block_size = 128 

n_layer = 8
n_head = 7
n_embd = 350

dropout = 0.2
weight_decay = 0.06

learning_rate = 2.5e-4  
max_iters = 12000
lr_decay_iters = 12000
min_lr = 2e-5

beta2 = 0.99

dtype = 'float16'

warmup_iters = 200