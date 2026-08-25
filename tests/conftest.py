"""Shared fixtures. OWNER: Person A.

    ./.venv/Scripts/python.exe -m pytest -q
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pemwe import load_config, StackModel, DegradationModel, PEMWEEnv  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def stack(cfg):
    return StackModel(cfg)


@pytest.fixture(scope="session")
def deg(cfg):
    return DegradationModel(cfg)


@pytest.fixture
def env(cfg):
    return PEMWEEnv(cfg, seed=0)
