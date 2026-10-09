"""Two processes, one set of fits and predicts: smoke_live.py twice on one cache dir.

The first process misses the cache: it fetches freMTPL2 from OpenML, fits
the three models (three POSTs to a fit endpoint) and predicts, and stores the
sources, the models, the staged calibration slice and the predictions. The
second reads them back and posts nothing. Live: run under direnv.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from post_counter import MODEL_NAMES


REPO = Path(__file__).resolve().parent.parent


def run_once(cache_dir: str) -> dict[str, str]:
    env = os.environ | {"XORQ_CACHE_DIR": cache_dir}
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_live.py"],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    # Print whatever the child produced before judging it, so a failed
    # assertion inside smoke_live.py still shows the figures it was judging.
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr[-3000:], end="", file=sys.stderr)
        raise SystemExit(f"smoke_live.py exited {proc.returncode}")
    return dict(re.findall(r"(\w+)=(\S+)", proc.stdout))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xorq-tabpfn-cache-") as cache_dir:
        print(f"cache_dir={cache_dir}")
        first = run_once(cache_dir)
        print("---")
        second = run_once(cache_dir)
    assert first["fit_posts"] == str(len(MODEL_NAMES)), first
    assert int(first["predict_posts"]) > 0, first
    assert second["fit_posts"] == "0", second
    assert second["predict_posts"] == "0", second
    for name in MODEL_NAMES:
        assert first[f"{name}_id"] == second[f"{name}_id"], (first, second)
    assert first["premium_calibrated_gini"] == second["premium_calibrated_gini"]
    print("three fits across two processes, then none: ok")


if __name__ == "__main__":
    main()
