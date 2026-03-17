"""
Sample from a trained model
"""
import os
import pickle
import json
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "FILE:data/rocstories/eval_prompts.txt" # Prompt. Can also specify a file, use as: "FILE:prompt.txt"
batch_prompts = True # if True, read multiple prompts from the file (one per line)
output_file = 'samples.jsonl' # file to save generated samples in JSONL format (set to None to disable)
num_samples = 1 # number of samples to generate for each prompt
max_new_tokens = 512 # number of tokens generated in each sample``
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if os.path.exists(os.path.join(out_dir, 'sample_params.json')):
        with open(os.path.join(out_dir, 'sample_params.json'), 'rb') as f:
            sample_params = json.load(f)
    else:
        sample_params = {
            'temperature': 0.8,
            'top_k': 200
        }
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    sample_params = {
        'temperature': 0.8,
        'top_k': 200
    }

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        if batch_prompts:
            # Read multiple prompts from file (one per line)
            prompts = [line.rstrip() for line in f.readlines()]
        else:
            # Read single prompt from file
            prompts = [f.read()]
else:
    prompts = [start]

# Encode all prompts
start_ids_list = [encode(prompt) for prompt in prompts]

# Create tensor from all prompts
x_list = [torch.tensor(ids, dtype=torch.long, device=device) for ids in start_ids_list]

# Open output file if specified
output_f = None
if output_file:
    output_f = open(output_file, 'w', encoding='utf-8')

# run generation
with torch.no_grad():
    with ctx:
        for prompt_idx, x_single in enumerate(x_list):
            x = x_single[None, ...]
            prompt_text = prompts[prompt_idx]
            if batch_prompts and len(prompts) > 1:
                prompt_header = f"\n=== Prompt {prompt_idx + 1}: {prompt_text} ==="
                print(prompt_header)
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, **sample_params)
                sample_text = decode(y[0].tolist())
                # Delete context after <|endoftext|>
                if '<|endoftext|>' in sample_text:
                    sample_text = sample_text.split('<|endoftext|>')[0]
                print(sample_text)
                print('---------------')
                
                # Save to JSONL file if specified
                if output_f:
                    record = {
                        # 'prompt_idx': prompt_idx,
                        'prompt': prompt_text,
                        # 'sample_idx': k,
                        'generated_text': sample_text,
                        'params': {
                            'max_new_tokens': max_new_tokens,
                            **sample_params
                        }
                    }
                    output_f.write(json.dumps(record) + '\n')

# Close output file if opened
if output_f:
    output_f.close()
    print(f"\nResults saved to {output_file}")
