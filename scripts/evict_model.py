"""Drop the cached model references and what was predicted with them.

Each cached "model" is a server-side model_id_ (docs/internals.md, "What
TabPFN returns from fit"); once the server has dropped the training data behind one, every
cache hit raises FittedModelNotFoundError at predict time and the recovery is
a re-fit. This removes exactly the entries expr.py's ``model_state`` names,
the three models, the staged calibration slice and the predictions, from the
store XORQ_CACHE_DIR resolves to; the OpenML sources and the ordered slices
stay. Restart a running serve_flight.py afterwards: its pinned reads have no
fallback, and the restart pins again and fits again.

By default the keys are the ones serve_flight.py recorded in the store's
MODEL_STATE_RECORD when it pinned, so nothing is fetched from anyone, and the
entries dropped are the ones that server reads even if OpenML's headers have
changed since. ``--from-expr`` computes the keys from expr.py instead, for a
store no server has recorded (smoke_live.py, smoke_cache.py): that sends
OpenML a HEAD request per source and finds the entries only while OpenML's
headers are unchanged.

    direnv exec . python scripts/evict_model.py [--from-expr]
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from xorq.caching import ParquetCache


EXPR = Path(__file__).resolve().parent.parent / "expr.py"


def evict(cache: ParquetCache, keys: list[str]) -> dict[str, str]:
    """Drop each key's entry; "dropped" or "absent" per key."""
    outcome = {}
    for key in keys:
        # is_present, not exists: a corrupt entry is still one to drop.
        if cache.storage.is_present(key):
            cache.storage.drop(key)
            outcome[key] = "dropped"
        else:
            outcome[key] = "absent"
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-expr", action="store_true")
    args = parser.parse_args()

    # Loading expr.py reads nothing; only --from-expr's key computation does.
    ns = runpy.run_path(str(EXPR), run_name="evict")
    cache = ns["CACHE"]
    record = cache.storage.path / ns["MODEL_STATE_RECORD"]
    if args.from_expr:
        keys = [cache.calc_key(table) for table in ns["model_state"]]
    elif record.exists():
        keys = json.loads(record.read_text())
    else:
        raise SystemExit(
            f"no {record}: no server has pinned this store; "
            "pass --from-expr to compute the keys (HEADs OpenML)"
        )
    print(
        f"cache_dir={cache.storage.path} keys_from={'expr' if args.from_expr else record}"
    )
    for key, what in evict(cache, keys).items():
        print(f"{key}={what}")


if __name__ == "__main__":
    main()
