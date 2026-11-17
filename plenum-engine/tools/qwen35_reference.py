"""An independent check of the qwen35 (Qwen3.5 / Qwen3.8 dense) layers against Hugging Face.

Builds a tiny Qwen3_5ForCausalLM with random weights in transformers, runs it in float32,
writes the same weights as a GGUF the way llama.cpp's conversion lays them out, and
saves the reference logits of the last token:

    python plenum-engine/tools/qwen35_reference.py <out dir>
    cargo run --release --example logits -- <out dir>/tiny_qwen35.gguf <out dir>/rust.f32 <tokens>
    python plenum-engine/tools/qwen35_reference.py <out dir> --compare <out dir>/rust.f32

What the conversion changes, and the engine reads as stored (c/qwen36.c, GGUF mode):
the zero-centred RMSNorm weights are written as 1 + w; A_log is written as
A = -exp(A_log); and the DeltaNet value heads are "tiled": HF value head h, which reads
key head h // rep, goes to position (h % rep) * key_heads + h // rep, so that position p
reads key head p % key_heads.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

TOKENS = [3, 17, 42, 8, 29, 11, 5, 33, 2, 40]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

KH, VH, KD, VD, KN = 2, 6, 16, 16, 4  # 3 value heads per key head: the tiling matters
cfg = Qwen3_5TextConfig(
    vocab_size=64, hidden_size=96, intermediate_size=160, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
    linear_conv_kernel_dim=KN, linear_key_head_dim=KD, linear_value_head_dim=VD,
    linear_num_key_heads=KH, linear_num_value_heads=VH,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    tie_word_embeddings=False,
)
torch.manual_seed(1234)
model = Qwen3_5ForCausalLM(cfg).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            p.copy_(torch.log(torch.rand_like(p) * 4 + 0.5))
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "norm" in name:
            p.copy_(torch.randn_like(p) * 0.2)  # zero-centred norms: 1 + this; the gated norm: this
        else:
            p.copy_(torch.randn_like(p) * 0.15)

if "--compare" in sys.argv:
    want = np.fromfile(out / "reference.f32", dtype="<f4")
    got = np.fromfile(sys.argv[sys.argv.index("--compare") + 1], dtype="<f4")
    diff = np.abs(want - got).max()
    print(f"max |rust - transformers| = {diff:.3e} on logits up to {np.abs(want).max():.3f};"
          f" argmax rust {got.argmax()} transformers {want.argmax()}")
    sys.exit(0 if diff < 1e-3 * max(1.0, np.abs(want).max()) else 1)

with torch.no_grad():
    ids = torch.tensor([TOKENS])
    logits = model(input_ids=ids, position_ids=torch.arange(len(TOKENS))[None]).logits[0, -1]
logits.numpy().astype("<f4").tofile(out / "reference.f32")

# ---- the GGUF, as llama.cpp lays out a qwen35 file ----
rep = VH // KH
perm = [(h % rep) * KH + h // rep for h in range(VH)]  # HF head h -> file position perm[h]
order = np.argsort(perm)  # file position p holds HF head order[p]


def heads(x, dim, axis=0):
    """Reorders the value-head blocks (of `dim` rows or entries) along `axis`."""
    x = np.asarray(x)
    shp = list(x.shape)
    shp[axis : axis + 1] = [VH, dim]
    y = x.reshape(shp)
    y = np.take(y, order, axis=axis)
    return y.reshape(x.shape)


sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
T = {}
T["token_embd.weight"] = sd["model.embed_tokens.weight"]
T["output.weight"] = sd["lm_head.weight"]
T["output_norm.weight"] = 1 + sd["model.norm.weight"]
key_tot = KH * KD
for i, kind in enumerate(cfg.layer_types):
    p, b = f"model.layers.{i}.", f"blk.{i}."
    T[b + "attn_norm.weight"] = 1 + sd[p + "input_layernorm.weight"]
    T[b + "post_attention_norm.weight"] = 1 + sd[p + "post_attention_layernorm.weight"]
    for n in ("gate", "up", "down"):
        T[b + f"ffn_{n}.weight"] = sd[p + f"mlp.{n}_proj.weight"]
    if kind == "full_attention":
        a = p + "self_attn."
        T[b + "attn_q.weight"] = sd[a + "q_proj.weight"]
        T[b + "attn_k.weight"] = sd[a + "k_proj.weight"]
        T[b + "attn_v.weight"] = sd[a + "v_proj.weight"]
        T[b + "attn_output.weight"] = sd[a + "o_proj.weight"]
        T[b + "attn_q_norm.weight"] = 1 + sd[a + "q_norm.weight"]
        T[b + "attn_k_norm.weight"] = 1 + sd[a + "k_norm.weight"]
    else:
        a = p + "linear_attn."
        qkv = sd[a + "in_proj_qkv.weight"]
        T[b + "attn_qkv.weight"] = np.concatenate([qkv[: 2 * key_tot], heads(qkv[2 * key_tot :], VD)])
        T[b + "attn_gate.weight"] = heads(sd[a + "in_proj_z.weight"], VD)
        T[b + "ssm_beta.weight"] = heads(sd[a + "in_proj_b.weight"], 1)
        T[b + "ssm_alpha.weight"] = heads(sd[a + "in_proj_a.weight"], 1)
        conv = sd[a + "conv1d.weight"][:, 0, :]  # [channels, kernel]
        T[b + "ssm_conv1d.weight"] = np.concatenate([conv[: 2 * key_tot], heads(conv[2 * key_tot :], VD)])
        T[b + "ssm_dt.bias"] = heads(sd[a + "dt_bias"], 1)
        T[b + "ssm_a"] = heads(-np.exp(sd[a + "A_log"]), 1)
        T[b + "ssm_norm.weight"] = sd[a + "norm.weight"]  # the gated norm is not zero-centred
        T[b + "ssm_out.weight"] = heads(sd[a + "out_proj.weight"], VD, axis=1)


def s(x):
    e = x.encode()
    return struct.pack("<Q", len(e)) + e


A = "qwen35"
kv = [
    ("general.architecture", 8, A), ("general.name", 8, "tiny qwen35"),
    (f"{A}.block_count", 4, cfg.num_hidden_layers), (f"{A}.context_length", 4, 4096),
    (f"{A}.embedding_length", 4, cfg.hidden_size), (f"{A}.feed_forward_length", 4, cfg.intermediate_size),
    (f"{A}.attention.head_count", 4, cfg.num_attention_heads), (f"{A}.attention.head_count_kv", 4, cfg.num_key_value_heads),
    (f"{A}.attention.key_length", 4, cfg.head_dim), (f"{A}.attention.value_length", 4, cfg.head_dim),
    (f"{A}.attention.layer_norm_rms_epsilon", 6, cfg.rms_norm_eps), (f"{A}.rope.freq_base", 6, 10000.0),
    (f"{A}.rope.dimension_count", 4, int(cfg.head_dim * 0.25)), (f"{A}.full_attention_interval", 4, 4),
    (f"{A}.ssm.conv_kernel", 4, KN), (f"{A}.ssm.state_size", 4, KD), (f"{A}.ssm.group_count", 4, KH),
    (f"{A}.ssm.time_step_rank", 4, VH), (f"{A}.ssm.inner_size", 4, VH * VD),
]
head = b"GGUF" + struct.pack("<IQQ", 3, len(T), len(kv))
for k, t, v in kv:
    head += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<I" if t == 4 else "<f", v))
off, blobs = 0, []
for name, arr in T.items():
    arr = np.ascontiguousarray(arr, dtype="<f4")
    dims = list(arr.shape[::-1])  # ggml order: the row length first
    head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
    head += struct.pack("<IQ", 0, off)
    blobs.append(arr.tobytes())
    off = (off + len(blobs[-1]) + 31) // 32 * 32
with open(out / "tiny_qwen35.gguf", "wb") as f:
    f.write(head + b"\0" * ((-len(head)) % 32))
    for blob in blobs:
        f.write(blob + b"\0" * ((-len(blob)) % 32))
print("tokens:", " ".join(map(str, TOKENS)))
print("wrote", out / "tiny_qwen35.gguf", "and", out / "reference.f32", f"(argmax {int(logits.argmax())})")
