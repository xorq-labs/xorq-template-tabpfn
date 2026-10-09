"""Live: one Flight server scores new rows for two token-less clients.

With the token: the server starts with TABPFN_TOKEN in its environment
(inherited from direnv) and a fresh cache directory; two client processes
run with the token unset, each sending the policy features of a different
slice of the held-out split, and each gets premiums back. The server echoes
every POST to Prior Labs. The server pins at startup, so the three fits,
one per model, and the calibration slice's predicts happen before it
serves; the exchanges are predicts only.

Without the token: a second server starts WITHOUT the token on the first
run's now-warm cache, so it pins without fitting and serves, and the
client's exchange must fail at predict, with the server's no-token error
quoted. ``--without-token-only`` runs it alone, on a fresh cache: there the
server cannot fit, so it must exit during the pin with that error instead.
Run under direnv; the run with the token is three fits plus the predicts
for the calibration slice and the two exchanges, the run without it only a
reachability check.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from post_counter import FIT_ENDPOINTS, MODEL_NAMES


REPO = Path(__file__).resolve().parent.parent
SLICES = ((0, 40), (40, 40))
# One predict POST per record batch, and expr.py's whole_batches gives the
# predicted slices batches of 8192 rows. Staging predicts the ~10k-row calibration slice
# with both stage-one models, two batches each; every 40-row exchange is one
# batch through all three models: 10 = 4 + 3 x 2 in all.
STAGING_PREDICTS = 4
PREDICTS_PER_EXCHANGE = 3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class Server:
    """A serve_flight.py process whose stdout is drained on a thread.

    The server prints during exchanges (one line per POST), so leaving the
    pipe unread until the end would block it once the buffer filled. The
    same thread spots the ``serving`` line, so readiness is an Event with a
    deadline rather than a blocking readline that no deadline can interrupt.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.lines: list[str] = []
        self.ready = threading.Event()
        self.drain = threading.Thread(target=self._drain, daemon=True)
        self.drain.start()

    def _drain(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line)
            if line.startswith("serving "):
                self.ready.set()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready.wait(1):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early: {self.output()}")
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("server did not report ready in time")
        print(
            f"server: {next(ln for ln in self.lines if ln.startswith('serving ')).strip()}"
        )

    def output(self) -> str:
        self.drain.join(timeout=30)
        return "".join(self.lines)

    def stop(self) -> str:
        # terminate, then kill: a server stuck in a native gRPC call past the
        # grace period must not outlive the smoke holding the port and the token.
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.output()


def start_server(port: int, with_token: bool, cache_dir: str, timeout: float) -> Server:
    env = dict(os.environ, XORQ_CACHE_DIR=cache_dir)
    if not with_token:
        env.pop("TABPFN_TOKEN", None)
    proc = subprocess.Popen(
        [sys.executable, "scripts/serve_flight.py", "--port", str(port)],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = Server(proc)
    server.wait_ready(timeout)
    return server


def run_client(
    port: int, offset: int, limit: int, cache_dir: str
) -> subprocess.CompletedProcess:
    # The server's cache directory: the client reads the OpenML sources to
    # pick its rows, and they land where the server will read them, not in
    # the developer's shared store.
    env = dict(os.environ, XORQ_CACHE_DIR=cache_dir)
    env.pop("TABPFN_TOKEN", None)
    return subprocess.run(
        [
            sys.executable,
            "scripts/flight_client.py",
            "--port",
            str(port),
            "--offset",
            str(offset),
            "--limit",
            str(limit),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-token-only", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as cache_dir:
        if not args.without_token_only:
            assert "TABPFN_TOKEN" in os.environ, (
                "run under direnv: the server needs the token"
            )
            with_token(cache_dir)
        without_token(cache_dir, warm=not args.without_token_only)


def with_token(cache_dir: str) -> None:
    port = free_port()
    t0 = time.perf_counter()
    # A cold start fits three models and predicts the calibration slice.
    server = start_server(port, with_token=True, cache_dir=cache_dir, timeout=600)
    print(f"server_start_s={time.perf_counter() - t0:.1f}")
    try:
        clients = []
        for offset, limit in SLICES:
            t0 = time.perf_counter()
            client = run_client(port, offset, limit, cache_dir)
            clients.append((client, time.perf_counter() - t0))
    finally:
        server_output = server.stop()
    for (client, wall), (_, limit) in zip(clients, SLICES):
        print(client.stdout, end="")
        assert client.returncode == 0, client.stderr[-2000:]
        assert f"client_rows={limit}" in client.stdout, client.stdout
        print(f"client_run_s={wall:.1f}")
    posts = Counter(
        ln.split("=", 1)[1]
        for ln in server_output.splitlines()
        if ln.startswith("tabpfn_post=")
    )
    fits = sum(posts[endpoint] for endpoint in FIT_ENDPOINTS)
    print(f"fit_posts={fits} predict_posts={posts['predict']} posts={dict(posts)}")
    # One fit per model is the property under test; the exact predict count
    # pins the batching, so a change that multiplies the POSTs toward the
    # rate limit fails here rather than as a 429.
    assert fits == len(MODEL_NAMES), server_output[-3000:]
    predicts = STAGING_PREDICTS + PREDICTS_PER_EXCHANGE * len(SLICES)
    assert posts["predict"] == predicts, server_output[-3000:]
    keys = [
        ln for ln in server_output.splitlines() if ln.startswith("model_state_key=")
    ]
    print(
        f"model_state_keys={len(keys)} pinned={sum('pinned=True' in k for k in keys)}"
    )


def without_token(cache_dir: str, warm: bool) -> None:
    port = free_port()
    if warm:
        # The models are cached, so the pin fits nothing and the server
        # serves; the exchange's predict is where the token is needed.
        server = start_server(port, with_token=False, cache_dir=cache_dir, timeout=120)
        try:
            client = run_client(port, *SLICES[0], cache_dir)
        finally:
            server_output = server.stop()
        assert client.returncode != 0, "client succeeded against a token-less server"
        last = (client.stderr.strip().splitlines() or ["<no stderr>"])[-1]
        print(f"without_token_client_error={last}")
    else:
        try:
            server = start_server(
                port, with_token=False, cache_dir=cache_dir, timeout=600
            )
        except RuntimeError as e:
            server_output = str(e)
        else:
            server.stop()
            raise AssertionError("a token-less server pinned a cold cache")
        print("without_token_client_error=<server exited during the pin>")
    # The server's own error: the no-token RuntimeError from tabpfn_client's
    # init(), which does not survive the trip to the client intact.
    server_errors = [ln for ln in server_output.splitlines() if "Error" in ln]
    print(
        f"without_token_server_error={server_errors[-1] if server_errors else '<none>'}"
    )
    assert any("RuntimeError" in ln for ln in server_errors), server_output[-3000:]
    print("token read in the worker, not the client: ok")


if __name__ == "__main__":
    main()
