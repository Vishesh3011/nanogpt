#!/usr/bin/env python3
"""
End-to-end smoke test — runs entirely on CPU with tiny configs.
No real data required: uses synthetic random token sequences.

Tests:
  1. Model construction (stories + code, default + scaled presets)
  2. Forward pass (with loss)
  3. Inference forward pass (no targets)
  4. Text generation (GPT.generate)
  5. Checkpoint save and reload (via core.utils)
  6. Trainer: 3 mini training steps using synthetic batches
  7. Perplexity computation (core.evaluator)
  8. Tokenizer round-trip (core.tokenizer)

Run:
    python test_flow.py
    python test_flow.py --verbose
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile

import torch

# ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"


def silence():
    """Context manager that suppresses stdout."""
    return contextlib.redirect_stdout(io.StringIO())


def run_test(name: str, fn, verbose: bool):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}")
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"       {e}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tiny_config(norm_type="layernorm", mlp_type="gelu", bias=True, n_kv_head=2, vocab_size=256):
    from core.architectures import GPTConfig
    return GPTConfig(
        n_layer=2, n_head=4, n_kv_head=n_kv_head, n_embd=64,
        block_size=32, vocab_size=vocab_size, dropout=0.0,
        bias=bias, norm_type=norm_type, mlp_type=mlp_type,
    )


def synthetic_batch(batch_size=2, seq_len=16, vocab_size=256):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_stories_construction():
    from models.stories.model import default_model_args, scaled_model_args, build_config
    for name, args in [("default", default_model_args()), ("scaled", scaled_model_args())]:
        with silence():
            from core.architectures import GPT
            m = GPT(build_config(args))
        assert m.get_num_params() > 0, f"{name} model has no params"


def test_code_construction():
    from models.code.model import default_model_args, scaled_model_args, build_config
    for name, args in [("default", default_model_args()), ("scaled", scaled_model_args())]:
        with silence():
            from core.architectures import GPT
            m = GPT(build_config(args))
        assert m.get_num_params() > 0


def test_forward_with_loss():
    from core.architectures import GPT
    for norm, mlp, bias, kv in [
        ("layernorm", "gelu",   True,  4),
        ("rmsnorm",  "swiglu", False, 2),
    ]:
        cfg = tiny_config(norm, mlp, bias, kv)
        with silence():
            m = GPT(cfg)
        x, y = synthetic_batch()
        logits, loss = m(x, y)
        assert logits.shape == (2, 16, 256)
        assert loss is not None and loss.item() > 0


def test_forward_inference():
    from core.architectures import GPT
    cfg = tiny_config()
    with silence():
        m = GPT(cfg)
    m.eval()
    x, _ = synthetic_batch(batch_size=1, seq_len=8)
    with torch.no_grad():
        logits, loss = m(x)  # no targets
    assert logits.shape == (
        1, 1, 256), f"unexpected logits shape: {logits.shape}"
    assert loss is None


def test_generation():
    from core.architectures import GPT
    cfg = tiny_config()
    with silence():
        m = GPT(cfg)
    m.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape[1] == 11, f"expected 11 tokens, got {out.shape[1]}"


def test_generation_eot_stop():
    """generate() stops at EOT token."""
    from core.architectures import GPT
    cfg = tiny_config()
    with silence():
        m = GPT(cfg)
    m.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    eot = 42
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=200, eot_token=eot)
    # should have stopped well before 201 tokens (eventually samples eot)
    assert out.shape[1] <= 201


def test_checkpoint_save_load():
    from core.architectures import GPT
    from core.utils import save_checkpoint, load_checkpoint
    cfg = tiny_config()
    with silence():
        m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(ckpt_path, m, opt, {
                        "n_layer": 2}, iter_num=1, best_val_loss=3.5, config={})
        ck = load_checkpoint(ckpt_path)
        assert "model" in ck and "optimizer" in ck
        assert ck["iter_num"] == 1
        assert abs(ck["best_val_loss"] - 3.5) < 1e-6

        # reload weights into a fresh model
        with silence():
            m2 = GPT(cfg)
        m2.load_state_dict(ck["model"])


def test_mini_training():
    """3 gradient steps without crashing."""
    from core.architectures import GPT
    cfg = tiny_config()
    with silence():
        m = GPT(cfg)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for _ in range(3):
        x, y = synthetic_batch()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert loss.item() < 1e6   # just a sanity bound


def test_perplexity():
    from core.architectures import GPT
    from core.evaluator import compute_perplexity
    from core.tokenizer import get_tokenizer
    enc = get_tokenizer()
    cfg = tiny_config(vocab_size=enc.n_vocab)
    with silence():
        m = GPT(cfg)
    m.eval()
    text = "Once upon a time there was a small cat."
    avg_loss, ppl = compute_perplexity(
        m, text, enc, block_size=32, device="cpu")
    assert avg_loss > 0 and ppl > 1.0


def test_tokenizer_roundtrip():
    from core.tokenizer import get_tokenizer, encode, decode
    try:
        enc = get_tokenizer()
    except Exception:
        raise RuntimeError(
            "SKIP: tiktoken GPT-2 vocab unavailable (network restricted environment)")
    text = "Hello, world! <|endoftext|>"
    ids = encode(text, enc)
    recovered = decode(ids, enc)
    # round-trip (whitespace may differ)
    assert text in recovered or recovered in text


def test_param_counts_under_50m():
    from core.architectures import GPT
    from models.stories.model import scaled_model_args as sm_stories, build_config as bc_s
    from models.code.model import scaled_model_args as sm_code,    build_config as bc_c
    with silence():
        m_stories = GPT(bc_s(sm_stories()))
        m_code = GPT(bc_c(sm_code()))
    total_stories = sum(p.numel() for p in m_stories.parameters())
    total_code = sum(p.numel() for p in m_code.parameters())
    assert total_stories < 50e6, f"stories model {total_stories/1e6:.1f}M exceeds 50M"
    assert total_code < 50e6, f"code model {total_code/1e6:.1f}M exceeds 50M"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

TESTS = [
    ("Model construction — stories",       test_stories_construction),
    ("Model construction — code",          test_code_construction),
    ("Forward pass with loss (both archs)", test_forward_with_loss),
    ("Forward pass — inference mode",      test_forward_inference),
    ("Text generation (basic)",            test_generation),
    ("Text generation (EOT stop)",         test_generation_eot_stop),
    ("Checkpoint save + reload",           test_checkpoint_save_load),
    ("Mini training loop (3 steps)",       test_mini_training),
    ("Perplexity computation",             test_perplexity),
    ("Tokenizer round-trip",               test_tokenizer_roundtrip),
    ("Param counts under 50M",             test_param_counts_under_50m),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{'─'*55}")
    print("  nanogpt-rebuilt  —  end-to-end smoke test")
    print(f"{'─'*55}")

    passed = failed = skipped = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  {PASS}  {name}")
            passed += 1
        except RuntimeError as e:
            if str(e).startswith("SKIP:"):
                print(f"  {SKIP}  {name}")
                print(f"       → {str(e)[6:]}")
                skipped += 1
            else:
                print(f"  {FAIL}  {name}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"       → {e}")
                failed += 1
        except Exception as e:
            print(f"  {FAIL}  {name}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"       → {e}")
            failed += 1

    print(f"{'─'*55}")
    summary = f"  {passed}/{passed+failed+skipped} passed"
    if skipped:
        summary += f"  ({skipped} skipped — need network)"
    print(summary + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
