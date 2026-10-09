"""Offline checks on expr.py's insurance chain and on what xorq pickles as "a model".

No request reaches Prior Labs or OpenML here. The chain is exercised through
expr.py's own builders with stand-in estimators and synthetic policies, and
the client's fitted state is assigned directly, which
``tabpfn_client/persistence.py`` documents as a supported way to make an
estimator fitted (``model_id_`` is the single source of truth for fittedness).
"""

from __future__ import annotations

import inspect
import json
import runpy
import socket
import threading
import time
import uuid
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.errors import FittedModelNotFoundError

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.api import _
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.ml.fit_lib import deferred_fit_predict_sklearn
from xorq.expr.relations import CacheTag, flight_serve_unbound
from xorq.flight import FlightUrl
from xorq.internal import SessionConfig


REPO = Path(__file__).resolve().parent.parent
SENTINEL = "SENTINEL-NOT-A-REAL-TOKEN"
STALE_MESSAGE = "stub: the server has no fitted model for this id"
P_CLAIM = 0.25
N_POLICIES = 400


@pytest.fixture(scope="module")
def ns(monkeypatch_module: pytest.MonkeyPatch) -> dict:
    # Loaded the way `xorq build` loads it, as __main__, with the network
    # cut: building the expression must read nothing, from OpenML or anyone.
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("loading expr.py touched the network")

    monkeypatch_module.setattr(socket.socket, "connect", refuse)
    monkeypatch_module.setattr(socket, "create_connection", refuse)
    namespace = runpy.run_path(str(REPO / "expr.py"), run_name="__main__")
    monkeypatch_module.undo()
    return namespace


@pytest.fixture(scope="module")
def monkeypatch_module() -> pytest.MonkeyPatch:
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small freMTPL2 look-alike: policies, and claims for some of them.

    It carries one of each case the cleaning handles: an exposure above a
    year, a claim above the amount cap, and a policy with several claims.
    """
    rng = np.random.default_rng(0)
    ids = np.arange(1, N_POLICIES + 1, dtype="float64")
    freq = pd.DataFrame(
        {
            "IDpol": ids,
            "ClaimNb": rng.integers(0, 2, N_POLICIES).astype("uint8"),
            "Exposure": rng.uniform(0.05, 1.0, N_POLICIES),
            "Area": rng.choice(list("ABCDEF"), N_POLICIES),
            "VehPower": rng.integers(4, 15, N_POLICIES).astype("uint8"),
            "VehAge": rng.integers(0, 20, N_POLICIES).astype("uint8"),
            "DrivAge": rng.integers(18, 90, N_POLICIES).astype("uint8"),
            "BonusMalus": rng.integers(50, 150, N_POLICIES).astype("uint8"),
            "VehBrand": rng.choice(["B1", "B2", "B12"], N_POLICIES),
            "VehGas": rng.choice(["Diesel", "Regular"], N_POLICIES),
            "Density": rng.uniform(1, 20_000, N_POLICIES),
            "Region": rng.choice(["R11", "R24", "R82"], N_POLICIES),
        }
    )
    freq.loc[0, "Exposure"] = 1.5
    claimants = ids[rng.uniform(size=N_POLICIES) < 0.3]
    sev = pd.DataFrame(
        {
            "IDpol": np.concatenate([claimants, claimants[:5]]),
            "ClaimAmount": rng.uniform(100, 5_000, len(claimants) + 5),
        }
    )
    sev.loc[0, "ClaimAmount"] = 500_000.0
    return freq, sev


@pytest.fixture(scope="module")
def policies(ns: dict, source_frames: tuple[pd.DataFrame, pd.DataFrame]) -> xo.Table:
    freq, sev = source_frames
    return ns["make_policies"](xo.memtable(freq), xo.memtable(sev))


class StandIn:
    """sklearn-shaped stand-in; no request leaves the process.

    ``fit`` mints a local uuid the way the client keeps a server-assigned one.
    Fits and predicts are recorded in ``CALLS`` as (kind, columns, frame).
    """

    def __init__(self, **params: object) -> None:
        self.params = params

    def get_params(self, deep: bool = True) -> dict:
        return dict(self.params)

    def set_params(self, **params: object) -> StandIn:
        self.params.update(params)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StandIn:
        CALLS.append(("fit", tuple(X.columns), X.assign(_y=np.asarray(y))))
        self.model_id_ = uuid.uuid4()
        self.classes_ = np.unique(y)
        self.mean_ = float(np.mean(y))
        return self


class StandInClassifier(StandIn):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        CALLS.append(("predict", tuple(X.columns), X))
        return np.tile([1 - P_CLAIM, P_CLAIM], (len(X), 1))


class StandInRegressor(StandIn):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        CALLS.append(("predict", tuple(X.columns), X))
        return np.full(len(X), self.mean_)


class SlowStandInClassifier(StandInClassifier):
    """A classifier fit slow enough for Flight exchanges to pile up behind it."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SlowStandInClassifier:
        time.sleep(1.0)
        return super().fit(X, y)


CALLS: list[tuple[str, tuple[str, ...], pd.DataFrame]] = []


def fits() -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    return [(cols, frame) for kind, cols, frame in CALLS if kind == "fit"]


def predicts() -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    return [(cols, frame) for kind, cols, frame in CALLS if kind == "predict"]


def stand_in_chain(
    ns: dict,
    policies: xo.Table,
    cache: ParquetCache | Path,
    classifier: type[StandIn] = StandInClassifier,
    whole_batches: bool = False,
) -> tuple:
    """expr.py's chain over synthetic policies, with stand-in estimators.

    ``cache`` is a store, or the root to make one at. The root is passed
    explicitly: xorq reads XORQ_CACHE_DIR once at import, so setting it here
    would land the stand-ins in the shared cache. ``whole_batches`` routes the
    calibration and test slices through expr.py's, as expr.py does.
    """
    if isinstance(cache, Path):
        cache = ParquetCache.from_kwargs(base_path=cache)
    fit, calibration, test = ns["make_slices"](policies, sample_fraction=1.0)
    if whole_batches:
        calibration, test = (ns["whole_batches"](t, cache) for t in (calibration, test))
    score, calibration_staged = ns["make_scorer"](
        fit,
        calibration,
        cache,
        classifier=classifier,
        regressor=StandInRegressor,
        classifier_params=ns["COMMON_PARAMS"],
        regressor_params=ns["AMOUNT_PARAMS"],
    )
    return cache, fit, calibration, test, score, calibration_staged


def policy_input(ns: dict, policies: xo.Table) -> xo.Table:
    """The unbound policy-feature table a served chain takes."""
    features = ns["POLICY_FEATURES"]
    return xo.table({f: policies.schema()[f] for f in features}, name="new")


def test_expr_uses_the_stock_estimators(ns: dict) -> None:
    defaults = inspect.signature(ns["make_scorer"]).parameters
    assert defaults["classifier"].default is TabPFNClassifier
    assert defaults["regressor"].default is TabPFNRegressor
    assert ns["TOKEN_ENV_VAR"] == "TABPFN_TOKEN"


def test_make_policies_joins_and_cleans(
    ns: dict, policies: xo.Table, source_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    freq, sev = source_frames
    df = policies.execute().set_index("IDpol").sort_index()
    totals = sev.groupby("IDpol").ClaimAmount.sum()
    expected = (
        freq.IDpol.map(totals).fillna(0.0).set_axis(freq.IDpol).sort_index()
    ).clip(upper=ns["CLAIM_AMOUNT_CAP"])
    assert len(df) == len(freq)
    pd.testing.assert_series_equal(df.ClaimAmount, expected, check_names=False)
    assert df.Exposure.max() == ns["EXPOSURE_CAP"]
    assert df.ClaimAmount.max() == ns["CLAIM_AMOUNT_CAP"]
    # A positive amount is a claim.
    assert (df[ns["CLAIM_TARGET"]] == (df.ClaimAmount > 0).astype("int64")).all()
    np.testing.assert_allclose(df.PurePremium, df.ClaimAmount / df.Exposure)


def test_make_slices_are_disjoint_and_sized(ns: dict, policies: xo.Table) -> None:
    whole = [set(t.execute().IDpol) for t in ns["make_slices"](policies, 1.0)]
    assert set.union(*whole) == set(policies.execute().IDpol)
    assert sum(map(len, whole)) == N_POLICIES
    half = [len(t.execute()) for t in ns["make_slices"](policies, 0.5)]
    assert 0.35 * N_POLICIES < sum(half) < 0.65 * N_POLICIES, half
    with pytest.raises(ValueError):
        ns["make_slices"](policies, 0.0)


def test_chain_fits_three_models_and_scores_at_a_full_year(
    ns: dict, policies: xo.Table, tmp_path: Path
) -> None:
    cache, fit, calibration, test, score, _staged = stand_in_chain(
        ns, policies, tmp_path
    )
    predicted = score(test.mutate(observed_exposure=_.Exposure)).cache(cache)
    fit_df, calibration_df = fit.execute(), calibration.execute()

    CALLS.clear()
    df = predicted.execute()
    features, calibration_features = ns["FEATURES"], ns["CALIBRATION_FEATURES"]
    assert [cols for cols, _ in fits()].count(features) == 2, fits()
    stage_one = [frame for cols, frame in fits() if cols == features]
    (calibration_fit,) = (f for cols, f in fits() if cols == calibration_features)
    # Claim model: every fit-slice policy, labelled 0/1. Severity model: the
    # claiming policies only, labelled with their amounts.
    (claim_fit,) = (f for f in stage_one if set(f._y) <= {0, 1})
    (severity_fit,) = (f for f in stage_one if f is not claim_fit)
    assert len(claim_fit) == len(fit_df)
    claimed = fit_df[fit_df[ns["CLAIM_TARGET"]] == 1]
    assert len(severity_fit) == len(claimed)
    np.testing.assert_allclose(
        np.sort(severity_fit._y), np.sort(claimed[ns["AMOUNT_TARGET"]])
    )
    # Calibration model: the calibration slice, its observed exposure, and
    # the two stage-one predictions made at a full year.
    assert len(calibration_fit) == len(calibration_df)
    np.testing.assert_allclose(
        np.sort(calibration_fit.Exposure), np.sort(calibration_df.Exposure)
    )
    assert (calibration_fit.p_claim == P_CLAIM).all()
    # Stage one only ever predicts at a full year.
    for cols, frame in predicts():
        if cols == features:
            assert set(frame.Exposure) == {1.0}
    assert len(df) == len(test.execute())
    assert (df.p_claim == P_CLAIM).all()
    np.testing.assert_allclose(df.premium, df.p_claim * df.severity)
    np.testing.assert_allclose(df.premium_calibrated, calibration_df.ClaimAmount.mean())
    assert (df.Exposure == 1.0).all()
    assert df.observed_exposure.notna().all()

    # A second execute is all cache hits: no fit, no predict.
    CALLS.clear()
    predicted.execute()
    assert CALLS == []


def test_model_state_is_every_model_dependent_entry(ns: dict) -> None:
    state = {table.op() for table in ns["model_state"]}
    # The three models, each one pickled-estimator column, and the two
    # entries predicted with them.
    models = [t for t in ns["model_state"] if t.schema().types == (dt.binary,)]
    assert len(models) == 3
    assert {ns["predicted"].op(), ns["calibration_staged"].op()} <= state
    assert len(state) == 5
    # Every cached node of the predictions is either model state or data no
    # model went into: the sources and the ordered slices.
    cached = set(ns["predicted"].ls.cached_nodes)
    data = {table.op() for table in ns["model_free"]}
    assert cached == state | data and not state & data


def test_amount_params_hash_and_reach_the_client_as_a_dict(ns: dict) -> None:
    params = dict(ns["AMOUNT_PARAMS"])
    hash(ns["AMOUNT_PARAMS"])
    model = TabPFNRegressor(**params)
    assert isinstance(model.get_params()["inference_config"], dict)


def test_fitted_model_pickle_round_trip_is_small(ns: dict) -> None:
    for cls, params in (
        (TabPFNClassifier, ns["COMMON_PARAMS"]),
        (TabPFNRegressor, ns["AMOUNT_PARAMS"]),
    ):
        model = cls(**dict(params))
        model.model_id_ = uuid.uuid4()
        blob = cloudpickle.dumps(model)
        restored = cloudpickle.loads(blob)
        assert len(blob) < 10_000
        assert restored.model_id_ == model.model_id_
        assert restored.get_params() == model.get_params()


def test_token_never_enters_the_pickle(
    ns: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", SENTINEL)
    model = TabPFNRegressor(**dict(ns["AMOUNT_PARAMS"]))
    model.model_id_ = uuid.uuid4()
    assert SENTINEL.encode() not in cloudpickle.dumps(model)


def test_save_model_record_reloads(ns: dict) -> None:
    model = TabPFNClassifier(**dict(ns["COMMON_PARAMS"]))
    model.model_id_ = uuid.uuid4()
    model.classes_ = np.array([0, 1])
    model._n_train_rows = 270
    record = model.save_model()
    assert set(record) == {
        "classes",
        "model_id",
        "n_train_rows",
        "params",
        "tabpfn_client_version",
        "task",
    }
    assert "client_options" not in record["params"]
    restored = type(model).load_model(record)
    assert type(restored) is type(model)
    assert restored.model_id_ == model.model_id_


class StaleReferenceEstimator(StandIn):
    """A predict that behaves like a stale ``model_id_``: what
    ``tabpfn_client/client.py`` raises on the prepare-test-set 404."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise FittedModelNotFoundError(STALE_MESSAGE)


def test_stale_model_error_reaches_the_caller_intact() -> None:
    table = xo.memtable(
        pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.1, 0.9, 0.3], "y": [0, 1, 0, 1]}
        )
    )
    (_, _, predict) = deferred_fit_predict_sklearn(
        expr=table,
        target="y",
        features=("a", "b"),
        cls=StaleReferenceEstimator,
        return_type=dt.float64,
    ).deferred_model_udaf_other
    predicted = table.mutate(predict.on_expr(table).name("predicted"))

    with pytest.raises(FittedModelNotFoundError) as info:
        predicted.execute()

    # Not wrapped or translated by the UDF or the arrow reader: the class,
    # the message and the absence of a chained cause all survive.
    assert type(info.value) is FittedModelNotFoundError
    assert str(info.value) == STALE_MESSAGE
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


def test_scored_takes_policy_features_only(ns: dict) -> None:
    (unbound,) = walk_nodes(ops.UnboundTable, ns["scored"])
    assert tuple(unbound.schema.names) == ns["POLICY_FEATURES"]
    assert "premium_calibrated" in ns["scored"].schema().names
    for column in ("Exposure", "ClaimAmount", ns["CLAIM_TARGET"]):
        assert column not in unbound.schema.names


def scored_stand_in(ns: dict, policies: xo.Table, tmp_path: Path) -> tuple:
    _cache, _fit, _calibration, test, score, _staged = stand_in_chain(
        ns, policies, tmp_path
    )
    new_rows = policy_input(ns, policies)
    return new_rows, score(new_rows), test.select(*ns["POLICY_FEATURES"])


def test_serving_scores_new_rows_with_three_fits(
    ns: dict, policies: xo.Table, tmp_path: Path
) -> None:
    # The fits are bound to their slices, the only unbound input is the
    # policy-feature table, and the server is in-process so CALLS sees it.
    new_rows, scored, held_out = scored_stand_in(ns, policies, tmp_path)
    CALLS.clear()
    server, do_exchange = flight_serve_unbound(
        scored, flight_url=FlightUrl(host="localhost")
    )
    try:
        # What the server demands of an exchange: the policy schema, exactly.
        assert server.exchangers[-1].schema_in_required == new_rows.schema()
        first = do_exchange(held_out.limit(5)).read_pandas()
        fits_after_first = len(fits())
        second = do_exchange(held_out.limit(7, offset=5)).read_pandas()
        calls_before = list(CALLS)
        # The client-side refusal; xorq raises a bare assert here and a
        # ValueError in its sibling check, so accept either and pin where it
        # happened instead: before the exchange opened.
        with pytest.raises((AssertionError, ValueError)):
            do_exchange(held_out.limit(3).mutate(ClaimAmount=xo.literal(0.0)))
        assert CALLS == calls_before
    finally:
        server.close()
    assert (len(first), len(second)) == (5, 7)
    assert fits_after_first == len(fits()) == 3
    assert (first.Exposure == 1.0).all() and first.premium_calibrated.notna().all()
    assert "ClaimAmount" not in first.columns


def test_concurrent_cold_exchanges_fit_each_model_once(
    ns: dict, policies: xo.Table, tmp_path: Path
) -> None:
    # Three clients open exchanges at once against a cold cache. xorq's
    # FlightServer runs do_exchange under one lock, so the first exchange
    # fits and the other two read the cached models: three fits, not nine.
    # The client inputs are materialised first because executing one Read
    # node from several threads is a separate client-side problem.
    _cache, _fit, _calibration, test, score, _staged = stand_in_chain(
        ns, policies, tmp_path, classifier=SlowStandInClassifier
    )
    new_rows = policy_input(ns, policies)
    held_out = test.select(*ns["POLICY_FEATURES"]).limit(15).execute()
    inputs = [xo.memtable(held_out.iloc[i * 5 : (i + 1) * 5]) for i in range(3)]

    CALLS.clear()
    rows: dict[int, int] = {}
    server, do_exchange = flight_serve_unbound(
        score(new_rows), flight_url=FlightUrl(host="localhost")
    )
    try:
        threads = [
            threading.Thread(
                target=lambda i=i: rows.__setitem__(
                    i, len(do_exchange(inputs[i]).read_pandas())
                )
            )
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.close()
    assert rows == {0: 5, 1: 5, 2: 5}, rows
    assert len(fits()) == 3, [cols for cols, _ in fits()]


def test_evict_drops_every_model_state_entry(
    ns: dict, policies: xo.Table, tmp_path: Path
) -> None:
    # After one execute every model-dependent entry exists; evicting the
    # recorded keys leaves none, and a second evict reports them absent. The
    # script is loaded the way expr.py is, not imported: scripts/ is not a
    # package.
    evict = runpy.run_path(str(REPO / "scripts" / "evict_model.py"), run_name="lib")[
        "evict"
    ]
    cache, _fit, _calibration, test, score, _staged = stand_in_chain(
        ns, policies, tmp_path
    )
    predicted = score(test).cache(cache)
    state = ns["model_state_of"](predicted, ())
    predicted.execute()
    assert all(cache.exists(e) for e in state)
    keys = ns["record_model_state"](cache, state)
    record = cache.storage.path / ns["MODEL_STATE_RECORD"]
    assert json.loads(record.read_text()) == keys

    assert evict(cache, keys) == dict.fromkeys(keys, "dropped")
    assert not any(cache.exists(e) for e in state)
    assert evict(cache, keys) == dict.fromkeys(keys, "absent")


def test_pinned_scored_serves_without_its_sources(
    ns: dict, source_frames: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    # What serve_flight.py does: pin with ensure_materialized, so the fits
    # happen at the pin, then serve the pinned expression. With the source
    # files deleted, an unpinned exchange could not compute its cache keys
    # (a local source is stat'd, an https one HEADed); the pinned one does
    # not need them, and fits nothing.
    con = xo.connect()
    cache = ParquetCache.from_kwargs(base_path=tmp_path / "cache", source=con)
    reads = []
    for name, frame in zip(("freq", "sev"), source_frames):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        reads.append(
            deferred_read_parquet(path=path, con=con, table_name=name).cache(cache)
        )
    policies = ns["make_policies"](*reads)
    _cache, _fit, calibration, test, score, _staged = stand_in_chain(
        ns, policies, cache, whole_batches=True
    )
    features = ns["POLICY_FEATURES"]
    scored = score(policy_input(ns, policies))
    held_out = test.select(*features).limit(5).execute()

    CALLS.clear()
    pinned = scored.ls.pin(ensure_materialized=True)
    assert len(fits()) == 3
    data = (*reads, calibration, test)
    keys = ns["record_model_state"](cache, ns["model_state_of"](scored, data))
    # Every entry is pinned, nested ones included; the record holds exactly
    # the pinned entries a model went into: three models and the staged slice.
    pinned_keys = {node.parent.name for node in walk_nodes(CacheTag, pinned)}
    data_keys = {cache.calc_key(table) for table in data}
    assert len(keys) == 4
    assert pinned_keys - data_keys == set(keys), (pinned_keys, data_keys, keys)

    for name in ("freq", "sev"):
        (tmp_path / f"{name}.parquet").unlink()
    CALLS.clear()
    server, do_exchange = flight_serve_unbound(
        pinned, flight_url=FlightUrl(host="localhost")
    )
    try:
        df = do_exchange(xo.memtable(held_out)).read_pandas()
    finally:
        server.close()
    assert fits() == []
    pd.testing.assert_frame_equal(df[list(features)], held_out, check_dtype=False)
    assert df.premium_calibrated.notna().all()
    with pytest.raises(FileNotFoundError):
        cache.calc_key(scored)


def test_whole_batches_keep_predicts_to_one_per_slice(
    ns: dict, source_frames: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    # xorq posts one predict per record batch. Through whole_batches a slice
    # below DataFusion's batch size reaches each model as one batch on a
    # many-partition session, so the whole chain is five predicts: claim and
    # severity for the calibration and test slices, and the calibration model
    # for the test slice. The partition count is pinned, not the host's, so
    # the fragmentation this guards against happens on any runner.
    con = xo.connect(SessionConfig().with_target_partitions(48))
    freq, sev = source_frames
    big_freq = pd.concat(
        [freq.assign(IDpol=freq.IDpol + i * N_POLICIES) for i in range(25)],
        ignore_index=True,
    )
    big_sev = pd.concat(
        [sev.assign(IDpol=sev.IDpol + i * N_POLICIES) for i in range(25)],
        ignore_index=True,
    )
    policies = ns["make_policies"](
        con.register(pa.Table.from_pandas(big_freq), table_name="freq_t"),
        con.register(pa.Table.from_pandas(big_sev), table_name="sev_t"),
    )
    cache = ParquetCache.from_kwargs(base_path=tmp_path, source=con)
    cache, _fit, _calibration, test, score, _staged = stand_in_chain(
        ns, policies, cache, whole_batches=True
    )
    CALLS.clear()
    df = score(test).cache(cache).execute()
    assert len(df) > 1_000
    assert len(predicts()) == 5, [len(frame) for _, frame in predicts()]


def test_exchange_merges_client_batches(
    ns: dict, policies: xo.Table, tmp_path: Path
) -> None:
    # xorq posts one predict per record batch, and an exchange's batches are
    # whatever the client streamed. A client sending its rows in many small
    # batches must still cost one predict per model, and get its rows back in
    # the order sent.
    new_rows, scored, held_out = scored_stand_in(ns, policies, tmp_path)
    sent = held_out.limit(60).to_pyarrow().cast(new_rows.schema().to_pyarrow())
    batches = sent.to_batches(max_chunksize=5)
    assert len(batches) == 12
    server, _do_exchange = ns["serve_scored"](
        scored, flight_url=FlightUrl(host="localhost")
    )
    try:
        # Warm the fits first, so only the exchange under test's predicts count.
        _do_exchange(held_out.limit(1)).read_pandas()
        CALLS.clear()
        _fut, out = server.client.do_exchange_batches(
            server.exchangers[-1].command,
            pa.RecordBatchReader.from_batches(sent.schema, batches),
        )
        df = out.read_pandas()
    finally:
        server.close()
    assert len(predicts()) == 3, [len(frame) for _, frame in predicts()]
    pd.testing.assert_frame_equal(
        df[list(sent.column_names)], sent.to_pandas(), check_dtype=False
    )


def test_merged_batches_recut_in_order(ns: dict) -> None:
    table = pa.table({"x": range(23)})
    reader = pa.RecordBatchReader.from_batches(
        table.schema, table.to_batches(max_chunksize=3)
    )
    merged = list(ns["merged_batches"](reader, rows=10))
    assert [b.num_rows for b in merged] == [10, 10, 3]
    assert pa.Table.from_batches(merged).equals(table)
