"""Send policy-feature rows to the Flight server and read premiums back.

The rows are a slice of expr.py's held-out split in policy-id order,
executed once here, with only the policy features sent: no exposure, no claim
amount. Executing the split reads the OpenML sources through the store
XORQ_CACHE_DIR resolves to (smoke_flight.py points it at the server's).
Refuses to run with TABPFN_TOKEN set, so a passing run shows the token was
read in the worker and not here. Checks the rows come back as sent, each
priced, and prints the mean calibrated premium, or the server's error
verbatim.

    direnv exec . env -u TABPFN_TOKEN python scripts/flight_client.py --port 8815 --offset 0 --limit 40
"""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.flight.client import FlightClient


EXPR = Path(__file__).resolve().parent.parent / "expr.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    assert "TABPFN_TOKEN" not in os.environ, "the client must not hold the token"

    ns = runpy.run_path(str(EXPR), run_name="client")
    features = list(ns["POLICY_FEATURES"])
    # Ordered, so --offset slices are disjoint and the same on every run.
    rows = (
        ns["test_data"]
        .order_by("IDpol")
        .limit(args.limit, offset=args.offset)
        .select(*features)
        .execute()
    )

    client = FlightClient(host=args.host, port=args.port)
    (_, rbr) = client.do_exchange("default", xo.memtable(rows))
    df = rbr.read_pandas()
    assert "ClaimAmount" not in df.columns, list(df.columns)
    # The rows sent, in the order sent: nothing dropped, duplicated or swapped.
    pd.testing.assert_frame_equal(df[features], rows, check_dtype=False)
    assert df.premium_calibrated.notna().all(), df.premium_calibrated.isna().sum()
    print(
        f"client_rows={len(df)} client_mean_premium={df.premium_calibrated.mean():.1f} "
        f"client_sent_columns={','.join(features)} client_token_in_env=False"
    )


if __name__ == "__main__":
    main()
