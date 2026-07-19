"""Load tabular agent modules without executing the torch-eager package __init__."""

import importlib
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tabular_modules():
    package = sys.modules.get("agents")
    if package is None:
        package = types.ModuleType("agents")
        package.__path__ = [str(ROOT / "agents")]
        package.__package__ = "agents"
        sys.modules["agents"] = package

    names = (
        "base",
        "q_learning",
        "r_learning",
        "smart_r",
        "relaxed_smart",
        "harmonic_r",
        "mab-epsilon",
        "mab-ucb",
        "oracle",
        "random_agent",
    )
    return {
        name: importlib.import_module(f"agents.{name}")
        for name in names
    }
