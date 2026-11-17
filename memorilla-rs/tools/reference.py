"""Reference outputs of Memorilla's own PyTorch MemoryModule, to check the Rust port.

Loads memorilla/memory.py as it is (only its small `memorilla.utils` import is stubbed, so
neither transformers nor vLLM is needed), runs a real checkpoint on fixed random inputs in
float32, and writes the inputs and the output as raw little-endian float32 files:

    python tools/reference.py <memorilla repo> <checkpoint dir> <out dir> [num_docs]
    memorilla-rs verify --memory <checkpoint dir> --docs <out>/docs.f32 \
        --question-file <out>/question.f32 --expect <out>/expect.f32
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

repo, checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
num_docs = int(sys.argv[4]) if len(sys.argv) > 4 else 37

utils = types.ModuleType("memorilla.utils")
utils.CHECKPOINT_FILE, utils.CONFIG_FILE = "memory.pt", "config.json"
utils.checkpoint_file = lambda path: Path(path) / "memory.pt" if Path(path).is_dir() else Path(path)
utils.load_memory_state = lambda path: torch.load(utils.checkpoint_file(path), map_location="cpu", weights_only=True)
sys.modules["memorilla"] = types.ModuleType("memorilla")
sys.modules["memorilla.utils"] = utils

spec = importlib.util.spec_from_file_location("memorilla.memory", repo / "memorilla" / "memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)

module = memory.MemoryModule.from_pretrained(checkpoint)
print("stored dtype:", next(module.parameters()).dtype, "| config:", module.config)
module = module.float()

generator = torch.Generator().manual_seed(1234)
docs = torch.nn.functional.normalize(torch.randn(1, num_docs, module.config.embedding_dim, generator=generator), dim=-1)
question = torch.nn.functional.normalize(torch.randn(1, module.config.embedding_dim, generator=generator), dim=-1)
with torch.no_grad():
    expect = module(docs, None, question)

out.mkdir(parents=True, exist_ok=True)
for name, tensor in (("docs", docs), ("question", question), ("expect", expect)):
    tensor.numpy().astype("<f4").tofile(out / f"{name}.f32")
print("wrote", num_docs, "documents and", tuple(expect.shape), "memory tokens to", out)
