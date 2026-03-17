from datasets import load_dataset

dataset = load_dataset("mintujupally/ROCStories")

test_data = dataset['test']
stories = [row["text"] for row in test_data]

with open("data/rocstories/eval_stories.txt", "w") as f:
    f.write("\n\n".join(stories))