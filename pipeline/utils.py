"""Shared helpers: per-step run metadata (used to auto-draft the methods section)."""
import json
import platform
from importlib import metadata
from pathlib import Path

TRACKED = ["scanpy", "anndata", "scvi-tools", "anthropic", "numpy", "pandas", "scipy", "snakemake"]


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def write_step_metadata(meta_dir, step: str, params: dict) -> None:
    """Record parameters and package versions for one pipeline step."""
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "params": params,
        "python": platform.python_version(),
        "versions": {p: pkg_version(p) for p in TRACKED},
    }
    (meta_dir / f"{step}.json").write_text(json.dumps(record, indent=2, default=str))


def read_all_metadata(meta_dir) -> dict:
    out = {}
    for f in sorted(Path(meta_dir).glob("*.json")):
        rec = json.loads(f.read_text())
        out[rec["step"]] = rec
    return out
