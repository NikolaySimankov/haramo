"""Adds ``suggest_power`` to optuna.Trial at package import.

``trial.suggest_power(name, min, max, base=2)`` is equivalent to
``base ** trial.suggest_int(name, min, max)`` — an integer-exponent
log-spaced grid. The underlying parameter stored in the study is the
exponent itself, so TPE / GridSampler / pruning operate on a clean
integer space.
"""
import optuna


def suggest_power(self, name: str, min: int, max: int, base: int = 2):
    if not isinstance(min, int) or not isinstance(max, int):
        raise TypeError(
            "suggest_power requires integer min/max exponents "
            f"(got min={min!r}, max={max!r})"
        )
    exponent = self.suggest_int(name, min, max)
    value = base ** exponent
    # TPE / GridSampler see the integer exponent (study.best_params).
    # The exponentiated value is recorded on user_attrs so downstream
    # consumers (e.g. magic_now's best-params TSV writer) can prefer it.
    # FrozenTrial replays don't support set_user_attr; guard accordingly.
    if hasattr(self, "set_user_attr"):
        try:
            self.set_user_attr(name, value)
        except Exception:
            pass
    return value


optuna.Trial.suggest_power = suggest_power
optuna.trial.FrozenTrial.suggest_power = suggest_power
