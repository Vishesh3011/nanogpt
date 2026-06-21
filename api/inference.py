import os
import torch
from typing import Optional
from core.architectures import GPT, GPTConfig
from core.utils import load_checkpoint
from core.tokenizer import get_tokenizer, encode, decode, get_eot_token_id

# module-level state — models live here after startup
_models: dict[str, Optional[GPT]] = {
    "stories": None,
    "code": None,
}
_enc = None  # shared tokenizer


def load_all_models() -> dict[str, bool]:
    """
    Called once at startup via FastAPI lifespan.
    Loads both models from checkpoint paths defined in environment variables:
        STORIES_CKPT_PATH  e.g. "checkpoints/stories/ckpt.pt"
        CODE_CKPT_PATH     e.g. "checkpoints/code/ckpt.pt"
    If a checkpoint path is missing or loading fails, that model is set to
    None and the API marks it as unavailable in /health — but doesn't crash.
    Returns {"stories": bool, "code": bool} indicating what loaded.
    """
    global _enc
    _enc = get_tokenizer()

    results = {}
    for name, env_var in [("stories", "STORIES_CKPT_PATH"), ("code", "CODE_CKPT_PATH")]:
        path = os.getenv(env_var)
        if not path or not os.path.exists(path):
            print(f"[inference] {name} checkpoint not found at '{path}' — skipping")
            results[name] = False
            continue
        try:
            checkpoint = load_checkpoint(path, map_location="cpu")
            config = GPTConfig(**checkpoint["model_args"])
            model = GPT(config)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            _models[name] = model
            print(f"[inference] {name} model loaded ({model.get_num_params()/1e6:.1f}M params)")
            results[name] = True
        except Exception as e:
            print(f"[inference] failed to load {name}: {e}")
            results[name] = False

    return results


def get_model(name: str) -> GPT:
    """Return a loaded model or raise RuntimeError if unavailable."""
    model = _models.get(name)
    if model is None:
        raise RuntimeError(f"Model '{name}' is not loaded. Check checkpoint path.")
    return model


def is_loaded(name: str) -> bool:
    return _models.get(name) is not None


def run_inference(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """
    Tokenize prompt → run model.generate() → decode output.
    For stories: uses stop_at_second_eot=True so output is a clean story.
    For code: truncates at </GO> tag if present.
    Raises RuntimeError if model not loaded.
    Raises ValueError if prompt tokenizes to more tokens than block_size.
    """
    model = get_model(model_name)
    enc = _enc

    token_ids = encode(prompt, enc)
    if len(token_ids) >= model.config.block_size:
        raise ValueError(
            f"Prompt is {len(token_ids)} tokens, exceeds model block_size "
            f"of {model.config.block_size}. Please shorten your prompt."
        )

    idx = torch.tensor([token_ids], dtype=torch.long)
    eot_id = get_eot_token_id(enc)

    with torch.no_grad():
        if model_name == "stories":
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eot_token=eot_id,
                stop_at_second_eot=True,
            )
        else:
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

    generated = decode(out[0].tolist(), enc)

    if model_name == "code" and "</GO>" in generated:
        generated = generated.split("</GO>")[0] + "</GO>"

    return generated.strip()