__version__ = "0.22.0"

from . import _optuna_patch  # noqa: F401 — installs trial.suggest_power

__all__ = [
    "classification",
    "regression",
    "utils",
    "varsel",
]
