"""Live smoke test: expr.py's insurance chain, three fits and their predicts.

Run under direnv (TABPFN_TOKEN comes from the secrets layer). Prints the
test-set size, the exposure-weighted Gini and the portfolio balance of the
uncalibrated and calibrated premiums, each model's reference and pickle size,
and how many POSTs went to each Prior Labs endpoint; never the token.

Everything is cached under XORQ_CACHE_DIR (docs/internals.md, "Caching the
fitted model"), so a second process pointed at the same directory posts nothing;
see ``smoke_cache.py`` for the two-process check.
"""

from __future__ import annotations

import pickle
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from post_counter import FIT_ENDPOINTS, MODEL_NAMES, PostCounter
from sklearn.metrics import auc

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt


EXPR = Path(__file__).resolve().parent.parent / "expr.py"


def gini_exposure(
    actual: pd.Series, predicted: pd.Series, exposure: pd.Series
) -> float:
    """The cookbook's ranking metric: 1 - 2 x area under the exposure Lorenz curve."""
    order = np.argsort(predicted.to_numpy())
    exp = exposure.to_numpy()[order]
    claim = actual.to_numpy()[order] * exp
    return 1 - 2 * auc(np.cumsum(exp) / exp.sum(), np.cumsum(claim) / claim.sum())


def balance(premium: pd.Series, df: pd.DataFrame) -> float:
    """Premium charged over the exposure observed, as a share of the claims paid."""
    return float((premium * df.observed_exposure).sum() / df.ClaimAmount.sum())


def model_name(model: xo.Table, ns: dict) -> str:
    """Which of the three fits a model node holds, from the columns it was fit on.

    A model node is one binary column named after the fit's inputs, the
    features and then the target: the claim model's target is the claim
    flag, the calibration model's features are the stage-one predictions.
    """
    (column,) = model.schema().names
    *features, target = column[column.index("(") + 1 : -1].split(", ")
    if target == ns["CLAIM_TARGET"]:
        return "claim_model"
    if tuple(features) == ns["CALIBRATION_FEATURES"]:
        return "calibration_model"
    return "severity_model"


ns = runpy.run_path(str(EXPR), run_name="smoke")
counter = PostCounter().attach()

t0 = time.perf_counter()
df = ns["predicted"].execute()
wall = time.perf_counter() - t0
print(f"rows={len(df)} claims={int(df.has_claim.sum())} fit+predict_s={wall:.1f}")
for column in ("premium", "premium_calibrated"):
    gini = gini_exposure(df.PurePremium, df[column], df.observed_exposure)
    print(f"{column}_gini={gini:.3f} {column}_balance={balance(df[column], df):.3f}")
fit_posts = sum(counter.posts.get(e, 0) for e in FIT_ENDPOINTS)
print(f"fit_posts={fit_posts} predict_posts={counter.posts.get('predict', 0)}")

# Each model node is the cached one-row table its predicts read, so these
# are cache hits after the execute above, not further fits.
before = dict(counter.posts)
models = {
    model_name(table, ns): table
    for table in ns["model_state"]
    if table.schema().types == (dt.binary,)
}
for name, model in models.items():
    blob = model.execute().iloc[0, 0]
    fitted = pickle.loads(blob)
    print(
        f"{name}_type={type(fitted).__name__} {name}_pickle_bytes={len(blob)} "
        f"{name}_id={fitted.model_id_}"
    )
model_posts = counter.since(before)
print(f"posts={dict(counter.posts)}")

# Checks last, so a failure still leaves the full receipt above.
assert len(df) > 0
assert sorted(models) == sorted(MODEL_NAMES), sorted(models)
assert df.premium_calibrated.notna().all(), df.premium_calibrated.isna().sum()
assert (df.Exposure == 1.0).all()
assert not model_posts, model_posts
