"""Shared path helpers for the dev-only golden-dump scripts.

Keeps the scripts portable across machines/users: output fixtures are derived from the repo
root (this file's location), never a hardcoded ``/Users/<name>`` path, and the model snapshot
honors the standard Hugging Face cache (``HF_HUB_CACHE`` / ``HF_HOME``, else
``~/.cache/huggingface/hub``).

The scripts are run directly (``python tools/dump_*.py``), so ``tools/`` is on ``sys.path`` and
``from _paths import fixture`` resolves.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/_paths.py -> the repo root is one directory up.
REPO_ROOT = Path(__file__).resolve().parents[1]


def fixture(rel: str) -> str:
    """Absolute path to a repo-relative output/fixture file.

    e.g. ``fixture("mlx-gen-z-image/tests/fixtures/z_latents.safetensors")``.
    """
    return str(REPO_ROOT / rel)


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache dir, honoring ``HF_HUB_CACHE`` / ``HF_HOME``.

    Falls back to the default ``~/.cache/huggingface/hub``.
    """
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def mflux_asset(name: str) -> str:
    """Absolute path to a bundled ``mflux`` asset (e.g. ``flux2_klein_edit.jpg``).

    Resolved from the installed ``mflux`` package via ``importlib.resources`` so it is portable
    across machines — never a hardcoded ``/Users/<name>`` path. Override the assets directory with
    ``MFLUX_ASSETS_DIR`` (e.g. a source checkout). Raises ``FileNotFoundError`` with an actionable
    message if the asset can't be located, rather than silently pointing at a missing absolute path.
    """
    if override := os.environ.get("MFLUX_ASSETS_DIR"):
        path = Path(override) / name
    else:
        try:
            from importlib.resources import files

            path = Path(str(files("mflux") / "assets" / name))
        except (ModuleNotFoundError, AttributeError) as exc:
            raise FileNotFoundError(
                f"mflux asset {name!r}: the 'mflux' package is not importable ({exc}). "
                "Run from the reference venv, or set MFLUX_ASSETS_DIR to the mflux assets dir."
            ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"mflux asset {name!r} not found at {path}. "
            "Set MFLUX_ASSETS_DIR to the directory that holds it."
        )
    return str(path)


def require_env(name: str, hint: str) -> str:
    """Value of the required env var ``name``, or a hard error with an actionable ``hint``.

    For machine-specific inputs that have no portable default (e.g. a reference-repo checkout path).
    Avoids baking a ``/Users/<name>`` default into the script.
    """
    if value := os.environ.get(name):
        return value
    raise SystemExit(f"{name} is required: {hint}")
