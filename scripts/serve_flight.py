"""Serve expr.py's ``scored`` over Arrow Flight; the token is read here.

The server process is the one doing the fit and predict, so TABPFN_TOKEN
has to be in ITS environment (run under direnv). The exchange's input is a
table of the nine policy-feature columns and nothing else: clients stream
policies in and get them back priced, with ``premium_calibrated`` last; they
need no token.

Before serving, ``scored`` is pinned (``ls.pin(ensure_materialized=True)``):
on a cold cache the three fits and the calibration slice's predicts happen
here, once, and each cached entry is then replaced by a plain read of its
file. An exchange is three predicts per 8192 rows, however the client
batched them (expr.py's ``serve_scored`` merges its batches), and computes no
cache key, so it never sends OpenML a HEAD request and a change on OpenML cannot
make a running server refit. A pinned read has no fallback: if its file is
evicted, exchanges fail until the server is restarted, which pins again
(and fits again). The keys of ``model_state``, computed at the same moment,
go to ``MODEL_STATE_RECORD`` in the store, which is what
``scripts/evict_model.py`` drops. Nothing serialises two server processes
on one cache directory: ``ParquetCache`` writes with tmp+rename and no lock,
so each process would fit each model once.

Each POST to Prior Labs is echoed to stdout as ``tabpfn_post=<endpoint>``
so a caller can count fits and predicts across exchanges; the endpoint is
whatever tabpfn-client posted to, so the async fit path (``submit_fit_job``)
shows up under its own name rather than vanishing.

    direnv exec . python scripts/serve_flight.py --port 8815
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
from pathlib import Path

from post_counter import PostCounter

from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import CacheTag
from xorq.flight import FlightUrl


EXPR = Path(__file__).resolve().parent.parent / "expr.py"


class PostEcho(PostCounter):
    """PostCounter that also prints one line per POST as it happens."""

    def emit(self, record: logging.LogRecord) -> None:
        if (endpoint := self.count(record)) is not None:
            print(f"tabpfn_post={endpoint}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    ns = runpy.run_path(str(EXPR), run_name="serve")
    PostEcho().attach()

    cache = ns["CACHE"]
    pinned = ns["scored"].ls.pin(ensure_materialized=True)
    keys = ns["record_model_state"](cache, ns["model_state"])
    # Every cached entry is pinned, the sources and ordered slices included.
    # The record must name every other one, or an eviction would leave a
    # file a model went into that this server reads.
    pinned_keys = {node.parent.name for node in walk_nodes(CacheTag, pinned)}
    data_keys = {cache.calc_key(table) for table in ns["model_free"]}
    assert pinned_keys - data_keys <= set(keys), (pinned_keys, data_keys, keys)
    for key in keys:
        print(f"model_state_key={key} pinned={key in pinned_keys}", flush=True)

    server, _ = ns["serve_scored"](
        pinned, flight_url=FlightUrl(host=args.host, port=args.port)
    )
    print(
        f"serving scored at grpc://{args.host}:{args.port} "
        f"input_columns={','.join(ns['POLICY_FEATURES'])} "
        f"token_in_env={'TABPFN_TOKEN' in os.environ}",
        flush=True,
    )
    server.wait()


if __name__ == "__main__":
    main()
