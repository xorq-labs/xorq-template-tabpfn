"""Live probe: which ``model_path`` names Prior Labs' API accepts for TabPFN 3.5.

``"v3.5_default"`` is an alias the server resolves to its current default
3.5 checkpoint, so a catalog entry that names it does not pin the weights.
Hugging Face publishes the 3.5 checkpoints under dated names
(``tabpfn-v3.5-20260909.safetensors``); this asks the API whether it accepts
those names in place of the alias. Each candidate gets one fit and one predict
on 32 synthetic rows with one estimator, for both estimator classes the chain
uses, and the script prints the API's answer and the prediction metadata it
returned (``billing_model_version`` and the ``model_path`` echoed back). A
made-up name is the negative control: if it is accepted too, acceptance means
nothing.

Run under direnv (TABPFN_TOKEN comes from the secrets layer); never prints the
token. It sends no policy data.

    direnv exec . python scripts/probe_model_path.py
"""

from __future__ import annotations

import numpy as np
from tabpfn_client import TabPFNClassifier, TabPFNRegressor


CANDIDATES = (
    "v3.5_default",  # control: the alias expr.py used before it pinned
    "tabpfn-v3.5-20260909",
    "tabpfn-v3.5-20260909.safetensors",
    "v3.5_20260909",
    "v3.5_no-such-checkpoint",  # negative control
)


def probe(estimator_cls: type, model_path: str, X: np.ndarray, y: np.ndarray) -> str:
    est = estimator_cls(model_path=model_path, n_estimators=1)
    try:
        est.fit(X, y)
        est.predict(X[:4])
    except RuntimeError as e:  # the client raises the API's 422 refusal as this
        return f"REFUSED  {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    meta = getattr(est, "_last_meta", None) or {}
    echoed = (meta.get("tabpfn_config") or {}).get("model_path")
    return (
        f"ACCEPTED billing_model_version={meta.get('billing_model_version')!r} "
        f"model_path echoed={echoed!r}"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3))
    targets = {
        TabPFNClassifier: (X[:, 0] > 0).astype(int),
        TabPFNRegressor: X @ np.array([1.0, -0.5, 0.25]),
    }
    for estimator_cls, y in targets.items():
        for model_path in CANDIDATES:
            result = probe(estimator_cls, model_path, X, y)
            print(f"{estimator_cls.__name__:17} {model_path:34} {result}")


if __name__ == "__main__":
    main()
