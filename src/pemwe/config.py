"""Config loading with CLI overrides. Shared by A, B and C -- change only by agreement.

Ablations are `--override reward.w2=2.0`, never an edited YAML file. On Day 5 you must be
able to reconstruct which config produced which number; edited files make that impossible.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CFG, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def config_id(cfg: dict, base: dict | None = None) -> str:
    """Short slug of what differs from default -- used to build run_ids."""
    base = base or load_config()
    diffs = []

    def walk(a, b, prefix=""):
        for k, v in a.items():
            if isinstance(v, dict):
                walk(v, b.get(k, {}), f"{prefix}{k}.")
            elif b.get(k) != v:
                diffs.append(f"{k}-{v}")

    walk(cfg, base)
    return "_".join(diffs) if diffs else "default"


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
