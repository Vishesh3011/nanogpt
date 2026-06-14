"""
Code-generation evaluation config.

Usage:
    python -m models.code.eval configs/code/eval_config.py
"""

out_dir = "checkpoints/code"
device = "cpu"
compile = False

input_file = "data/code/test.txt"

prompt = "<GO>\n# Function: add\n# Description: add two numbers\nfunc add(a int, b int) \n"
max_new_tokens = 80
temperature = 0.6
top_k = 40
