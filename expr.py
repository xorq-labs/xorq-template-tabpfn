"""Prior Labs' freMTPL2 insurance cookbook through xorq's deferred sklearn surface.

The source is Prior Labs' "Insurance Claim Modeling" cookbook
(docs.priorlabs.ai/cookbook/insurance_claim_modeling): French motor
third-party liability policies, priced as a pure premium, the expected claim
cost per year of exposure. The expression is the cookbook's two-stage TabPFN
model with a learned calibration model in place of its constant factor:

1. Two OpenML tables are read and joined: one row per policy (freMTPL2freq)
   and one row per claim (freMTPL2sev), the claims summed per policy.
2. A ``TabPFNClassifier`` predicts whether a policy claims at all, and a
   ``TabPFNRegressor`` trained on claiming policies only predicts the claim
   amount given a claim. Both are queried at a full year of exposure, and
   their product is the uncalibrated premium.
3. A second ``TabPFNRegressor`` is trained on those two predictions and the
   observed exposure of a calibration slice, and predicts the claim amount
   from them: the calibrated premium.

The calibration slice is held out of the stage-one fit because TabPFN
conditions on its training rows in context, so predictions on rows it was fit
on are too good to calibrate against (the cookbook makes the same point).

The estimators are the stock client classes: ``TABPFN_TOKEN`` is read by
tabpfn-client inside ``fit`` and ``predict``, never here, so it cannot end up
in a build. Each fitted model is
a reference to server-side state (docs/internals.md, "What TabPFN returns
from fit").
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import NamedTuple

import pyarrow as pa
from tabpfn_client import TabPFNClassifier, TabPFNRegressor

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.api import _
from xorq.caching import Cache, ParquetCache
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.rbr_utils import make_filtered_reader
from xorq.expr.ml.fit_lib import (
    deferred_fit_predict_proba_sklearn,
    deferred_fit_predict_sklearn,
)
from xorq.flight import FlightServer
from xorq.flight.exchanger import UnboundExprExchanger, streaming_expr_exchange
from xorq.vendor.ibis.common.collections import FrozenDict


TOKEN_ENV_VAR = "TABPFN_TOKEN"
MODEL_PATH = "v3.5_default"
N_ESTIMATORS = 8

OPENML_PARQUET_URL = "https://data.openml.org/datasets/0004/{id}/dataset_{id}.pq"
FREQ_ID = 41214
SEV_ID = 41215
# Given rather than inferred, so building the expression reads nothing; the
# string columns are dictionary-encoded in the files and read back as strings.
FREQ_SCHEMA = xo.schema(
    {
        "IDpol": "float64",
        "ClaimNb": "uint8",
        "Exposure": "float64",
        "Area": "string",
        "VehPower": "uint8",
        "VehAge": "uint8",
        "DrivAge": "uint8",
        "BonusMalus": "uint8",
        "VehBrand": "string",
        "VehGas": "string",
        "Density": "float64",
        "Region": "string",
    }
)
SEV_SCHEMA = xo.schema({"IDpol": "float64", "ClaimAmount": "float64"})

# What a policyholder is described by, and so what a Flight client sends.
POLICY_FEATURES = (
    "VehAge",
    "DrivAge",
    "VehBrand",
    "VehPower",
    "VehGas",
    "Region",
    "Area",
    "BonusMalus",
    "Density",
)
# Exposure is an input rather than a GLM offset, as in the cookbook; every
# prediction is made at a full year so the output is already per year.
FEATURES = (*POLICY_FEATURES, "Exposure")
CLAIM_TARGET = "has_claim"
AMOUNT_TARGET = "ClaimAmount"
CALIBRATION_FEATURES = ("p_claim", "severity", "Exposure")

# The cookbook's cleaning, from the scikit-learn tweedie example.
EXPOSURE_CAP = 1.0
CLAIM_AMOUNT_CAP = 200_000.0

# Share of the 678,013 policies used, and how that share is divided between
# the stage-one fit, the calibration fit and the test set. The default gives
# roughly 30k fit, 10k calibration and 10k test rows.
SAMPLE_FRACTION = 0.075
SLICE_SHARES = (0.6, 0.2, 0.2)
SPLIT_SEED = 42
NUM_BUCKETS = 100_000

COMMON_PARAMS = (
    ("model_path", MODEL_PATH),
    ("n_estimators", N_ESTIMATORS),
    ("ignore_pretraining_limits", True),
)
# Claim amounts are fit on log(1 + y), the one knob the cookbook turns. A
# FrozenDict rather than a dict because xorq hashes the params; it is a dict
# subclass, so the client sends it unchanged.
AMOUNT_PARAMS = (
    *COMMON_PARAMS,
    (
        "inference_config",
        FrozenDict({"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("1_plus_log",)}),
    ),
)

# Rows per batch a served exchange hands the chain: DataFusion's batch size,
# the size whole_batches gives the predicted slices.
EXCHANGE_BATCH_ROWS = 8192

# The file, in the store's directory, that a server writes the model_state
# keys it pinned to, and that scripts/evict_model.py reads.
MODEL_STATE_RECORD = "model_state.json"


def read_openml(
    dataset_id: int, schema: xo.Schema, cache: Cache, con: xo.Backend
) -> xo.Table:
    """One OpenML parquet file as a cached deferred read.

    The cache key of an https read is the URL plus the Last-Modified and
    Content-Length headers of a HEAD request, so a warm cache costs one HEAD
    and no download, and a changed file is fetched again.
    """
    return deferred_read_parquet(
        path=OPENML_PARQUET_URL.format(id=dataset_id),
        con=con,
        table_name=f"openml_{dataset_id}",
        schema=schema,
    ).cache(cache)


def make_policies(freq: xo.Table, sev: xo.Table) -> xo.Table:
    """One row per policy with its total claim amount, cleaned as the cookbook does."""
    claims = sev.group_by("IDpol").aggregate(ClaimAmount=_.ClaimAmount.sum())
    joined = freq.left_join(claims, "IDpol").drop("IDpol_right")
    amount = joined.ClaimAmount.fill_null(0.0)
    capped = joined.mutate(
        ClaimAmount=xo.least(amount, CLAIM_AMOUNT_CAP),
        Exposure=xo.least(_.Exposure, EXPOSURE_CAP),
    )
    return capped.mutate(
        PurePremium=_.ClaimAmount / _.Exposure,
        **{CLAIM_TARGET: (_.ClaimAmount > 0).cast("int64")},
    )


def make_slices(
    policies: xo.Table, sample_fraction: float = SAMPLE_FRACTION
) -> tuple[xo.Table, xo.Table, xo.Table]:
    """The fit, calibration and test slices of a sample, disjoint by policy id.

    One split call carves all three plus the unused remainder, so the slices
    cannot overlap and the sample size is a single number.
    """
    if not 0 < sample_fraction <= 1:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
    sizes = [sample_fraction * share for share in SLICE_SHARES]
    remainder = 1 - sum(sizes)
    fit, calibration, test, *_rest = xo.train_test_splits(
        policies,
        # A remainder of float dust is no remainder: the split checks the sum
        # to about seven places.
        test_sizes=[*sizes, remainder] if remainder > 1e-9 else sizes,
        unique_key="IDpol",
        num_buckets=NUM_BUCKETS,
        random_seed=SPLIT_SEED,
    )
    return fit, calibration, test


def whole_batches(slice_: xo.Table, cache: Cache) -> xo.Table:
    """A slice that a predict reads in full-size record batches on any host.

    xorq calls a predict UDF once per record batch, and each call is one POST
    to Prior Labs, which allows 60 predicts a minute. A join and split spread
    over one partition per CPU leave batches of a few hundred rows, so a
    many-core host would post hundreds of predicts, past the limit. Ordering
    merges the slice into one stream of 8192-row batches, and the cache keeps
    them as row groups, so the chain makes 10 predicts. It lives in the plan
    rather than in a session setting because `xorq build` records the
    connection without its config. Neither ordering nor caching does it alone.
    """
    return slice_.order_by("IDpol").cache(cache)


def make_scorer(
    fit: xo.Table,
    calibration: xo.Table,
    cache: Cache,
    classifier: type = TabPFNClassifier,
    regressor: type = TabPFNRegressor,
    classifier_params: tuple = COMMON_PARAMS,
    regressor_params: tuple = AMOUNT_PARAMS,
) -> tuple[Callable[[xo.Table], xo.Table], xo.Table]:
    """The chain as a function from a table of policies to the same rows scored.

    Also returns the staged calibration slice, the one cached table between the
    stage-one models and the calibration model.

    The estimator classes and params are arguments so the tests can run the
    same wiring with offline stand-ins. Every predict built by the returned
    function shares the three model nodes, so their cache keys are shared too:
    the test set, the calibration slice and a Flight exchange read one fit each.
    """
    # Each predict gets its own name_infix. xorq names a predict UDF from the
    # estimator class and params alone, so the severity and calibration
    # regressors, identical in both, would otherwise share one name, and the
    # backend would bind both calls to whichever signature it registered.
    claim_proba = deferred_fit_predict_proba_sklearn(
        expr=fit,
        target=CLAIM_TARGET,
        features=FEATURES,
        cls=classifier,
        return_type=dt.Array(dt.float64),
        params=classifier_params,
        name_infix="predict_proba_claim",
        cache=cache,
    ).deferred_other
    severity = deferred_fit_predict_sklearn(
        expr=fit.filter(_[CLAIM_TARGET] == 1),
        target=AMOUNT_TARGET,
        features=FEATURES,
        cls=regressor,
        return_type=dt.float64,
        params=regressor_params,
        name_infix="predict_severity",
        cache=cache,
    ).deferred_other

    def stage(policies: xo.Table) -> xo.Table:
        # Both stage-one predictions at a full year of exposure, side by side
        # in one mutate, and their product.
        unit = policies.mutate(Exposure=xo.literal(1.0, "float64"))
        return unit.mutate(
            # predict_proba gives one array per row in class order; the
            # classes are 0 and 1, so index 1 is P(claim).
            p_claim=claim_proba.on_expr(unit)[1],
            severity=severity.on_expr(unit),
        ).mutate(premium=_.p_claim * _.severity)

    # The calibration model learns from what stage one predicts at a full
    # year and from the exposure the slice was actually observed for, so its
    # target, the observed claim amount, is on the same footing.
    calibration_staged = (
        stage(calibration.mutate(observed_exposure=_.Exposure))
        .mutate(Exposure=_.observed_exposure)
        .drop("observed_exposure")
        .cache(cache)
    )
    calibrate = deferred_fit_predict_sklearn(
        expr=calibration_staged,
        target=AMOUNT_TARGET,
        features=CALIBRATION_FEATURES,
        cls=regressor,
        return_type=dt.float64,
        params=regressor_params,
        name_infix="predict_calibrated",
        cache=cache,
    ).deferred_other

    def score(policies: xo.Table) -> xo.Table:
        staged = stage(policies)
        return staged.mutate(premium_calibrated=calibrate.on_expr(staged))

    return score, calibration_staged


def model_state_of(expr: xo.Table, data: tuple[xo.Table, ...]) -> tuple[xo.Table, ...]:
    """Every cached entry inside ``expr`` that a fitted model went into.

    That is every cached node but ``data``, the entries no fit feeds: the
    models themselves, and everything predicted with them. Taken from the
    expression rather than rebuilt, because each construction of a
    fit-library model node carries a counter in its column name and so a new
    cache key (docs/internals.md, "Caching the fitted model").
    """
    skip = {table.op() for table in data}
    return tuple(node.to_expr() for node in expr.ls.cached_nodes if node not in skip)


def record_model_state(cache: Cache, state: tuple[xo.Table, ...]) -> list[str]:
    """Write the keys of ``state`` to the store's MODEL_STATE_RECORD.

    Computing a key hashes the whole upstream, the OpenML reads included, and
    an https read's hash is a HEAD request. Recorded once, the keys let an
    eviction drop exactly what was written, offline, however OpenML's
    headers have changed since.
    """
    keys = [cache.calc_key(table) for table in state]
    path = cache.storage.path / MODEL_STATE_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys, indent=2) + "\n")
    return keys


def merged_batches(
    reader: pa.RecordBatchReader, rows: int = EXCHANGE_BATCH_ROWS
) -> Iterator[pa.RecordBatch]:
    """The reader's rows, in order, re-cut into batches of ``rows``."""
    pending = pa.Table.from_batches([], schema=reader.schema)
    for batch in reader:
        pending = pa.concat_tables([pending, pa.Table.from_batches([batch])])
        while pending.num_rows >= rows:
            yield from pending.slice(0, rows).combine_chunks().to_batches()
            pending = pending.slice(rows)
    if pending.num_rows:
        yield from pending.combine_chunks().to_batches()


class _Chunk(NamedTuple):
    data: pa.RecordBatch


class _MergedReader:
    """A Flight exchange reader whose batches are merged_batches of the client's."""

    def __init__(self, reader: pa.flight.MetadataRecordBatchReader) -> None:
        self._batches = make_filtered_reader(reader)
        self.schema = self._batches.schema

    def __iter__(self) -> Iterator[_Chunk]:
        return (_Chunk(batch) for batch in merged_batches(self._batches))


class MergingExchanger(UnboundExprExchanger):
    """xorq's exchanger for an unbound expression, fed whole batches.

    xorq calls a predict UDF once per record batch, one POST each, and an
    exchange's batches are whatever the client streamed: 100 small batches
    would be 300 predicts against a limit of 60 a minute. Merging them here
    makes an exchange one predict per model per EXCHANGE_BATCH_ROWS rows,
    whatever the client sends.
    """

    @property
    def exchange_f(self) -> Callable:
        def exchange(context, reader, writer, **kwargs: object) -> None:
            streaming_expr_exchange(
                self.unbound_expr,
                self.make_connection,
                context,
                _MergedReader(reader),
                writer,
                **kwargs,
            )

        return exchange


def serve_scored(
    unbound_expr: xo.Table, **kwargs: object
) -> tuple[FlightServer, Callable]:
    """``flight_serve_unbound`` with a MergingExchanger: (server, do_exchange)."""
    exchanger = MergingExchanger(unbound_expr)
    server = FlightServer(**kwargs)
    server.exchangers += (exchanger,)
    server.serve()

    def do_exchange(expr: xo.Table) -> pa.RecordBatchReader:
        return server.client.do_exchange(exchanger.command, expr)[1]

    return server, do_exchange


CON = xo.connect()

# One store for the source files, the predicted slices, the three models,
# the staged calibration slice and the predictions. base_path is left unset
# so the store resolves to XORQ_CACHE_DIR at run time and `xorq run
# --cache-dir` can relocate it.
CACHE = ParquetCache.from_kwargs(source=CON)

freq = read_openml(FREQ_ID, FREQ_SCHEMA, CACHE, CON)
sev = read_openml(SEV_ID, SEV_SCHEMA, CACHE, CON)
policies = make_policies(freq, sev)
fit_data, calibration_data, test_data = make_slices(policies)
# The fit slice is one fit call per model, whatever its batches.
calibration_data = whole_batches(calibration_data, CACHE)
test_data = whole_batches(test_data, CACHE)
score, calibration_staged = make_scorer(fit_data, calibration_data, CACHE)

# The chain overwrites Exposure with a full year; the observed value rides
# along beside it, since the metrics weight by it.
predicted = score(test_data.mutate(observed_exposure=_.Exposure)).cache(CACHE)
# Everything in the store that a fitted model went into, which is what has
# to go when a model reference goes stale (scripts/evict_model.py): the
# three models, the staged calibration slice and the predictions. The
# source files and the ordered slices do not depend on Prior Labs.
model_free = (freq, sev, calibration_data, test_data)
model_state = model_state_of(predicted, model_free)

# Scoring policies the models never saw. The only unbound input is a table of
# the policy features; exposure is fixed at a year inside the chain, and the
# three fits stay bound to their slices.
new_rows = xo.table({f: policies.schema()[f] for f in POLICY_FEATURES}, name="new_rows")
scored = score(new_rows)

expr = predicted


# xorq's examples runner (python/xorq/tests/test_examples.py) runs a script
# under this name. The body is NOT offline-safe: on a cold cache it fetches
# freMTPL2 from OpenML and fits and predicts three models against Prior Labs,
# which needs TABPFN_TOKEN. Nothing in this repo invokes it.
if __name__ == "__pytest_main__":
    df = predicted.execute()
    assert df["premium_calibrated"].notna().all()
    pytest_examples_passed = True
