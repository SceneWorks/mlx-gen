"""Shared boilerplate for the SenseNova-U1 real-weight dump scripts (F-157).

Centralizes the HF snapshot resolution and the model + tokenizer load recipe that the five
``dump_sensenova_*_realweight.py`` scripts repeated verbatim, so a snapshot-pin or load-recipe
change is a one-line edit here instead of five. The snapshot now resolves through
``_paths.hf_hub_cache()`` (honoring ``HF_HUB_CACHE`` / ``HF_HOME``) — the hardcoded ``~/.cache``
literal the scripts used silently ignored that override.

Run from the SenseNova reference venv (``_vendor/sensenova_u1`` with ``PYTHONPATH=src``), like the
scripts that import this — ``tools/`` is on ``sys.path`` so ``from _sensenova_common import …``
resolves, mirroring ``_paths``.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from _paths import hf_hub_cache

# The SenseNova-U1-8B-MoT snapshot revision the committed goldens were dumped against.
SNAPSHOT_REV = "bfa9b436503cb8aed4f2bc60e3236710cc77468d"


def snapshot_dir() -> str:
    """Absolute path to the pinned model snapshot under the HF hub cache.

    Honors ``HF_HUB_CACHE`` / ``HF_HOME`` (via ``hf_hub_cache()``) instead of the hardcoded
    ``~/.cache/huggingface/hub`` literal (F-157). Raises ``FileNotFoundError`` if it is absent
    rather than failing later with a less obvious error.
    """
    path = hf_hub_cache() / "models--sensenova--SenseNova-U1-8B-MoT" / "snapshots" / SNAPSHOT_REV
    if not path.is_dir():
        raise FileNotFoundError(
            f"SenseNova snapshot not found at {path}. Download sensenova/SenseNova-U1-8B-MoT "
            f"(rev {SNAPSHOT_REV[:12]}…) or point HF_HUB_CACHE / HF_HOME at its cache."
        )
    return str(path)


def lora_glob(filename: str) -> str:
    """Glob pattern for ``filename`` under any snapshot of the SenseNova-U1-8B-MoT-LoRAs repo.

    Honors the HF cache env too (F-157); the fast dumper resolves the distill LoRA through this.
    """
    return str(
        hf_hub_cache()
        / "models--sensenova--SenseNova-U1-8B-MoT-LoRAs"
        / "snapshots"
        / "*"
        / filename
    )


def pick_device() -> str:
    """``"mps"`` when available, else ``"cpu"`` — the dumpers' device policy."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model_and_tokenizer(dtype: torch.dtype = torch.bfloat16, device: str | None = None):
    """Load ``(model, tokenizer, device)`` from the pinned snapshot at ``dtype``.

    ``dtype`` is the model compute dtype — ``bfloat16`` for most paths; the it2i path uses
    ``float32``. ``device`` defaults to :func:`pick_device`.
    """
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

    snap = snapshot_dir()
    device = device or pick_device()
    print(f"loading {snap} on {device} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    model = (
        NEOChatModel.from_pretrained(snap, torch_dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )
    return model, tok, device
