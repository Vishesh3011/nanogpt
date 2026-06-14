"""
Stories evaluation config.

Usage:
    python -m models.stories.eval configs/stories/eval_config.py
"""

init_from = "resume"
out_dir = "checkpoints/stories"
device = "cpu"
compile = False

input_file = "data/stories/eval_stories.txt"
input_format = "auto"
max_paragraphs = 200
print_first_n = 3

prompt = "\n"
max_new_tokens = 200
temperature = 0.8
top_k = 200