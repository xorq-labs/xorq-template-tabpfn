# How it works

What tabpfn-client and xorq do underneath `expr.py`, and the constraints
that shape the code. Versions: tabpfn-client 0.6.1, xorq 0.4.6. How to run
things is in the [README](../README.md). Each section starts with its
conclusion; the rest is the evidence, with the source file and function that
back it.

Terms used throughout:

- **expression**: xorq's description of the pipeline. Building one reads no
  data; work happens only when it is executed.
- **UDF**: a user-defined function xorq calls on the data. Each fit and each
  predict in the chain is one.
- **record batch**: a chunk of rows, the unit the query engine (DataFusion)
  passes around. xorq calls a predict UDF once per batch.
- **call to Prior Labs**: one HTTP POST to Prior Labs' API. Every fit and
  every predict UDF call is one, and Prior Labs allows 60 predicts a minute.
- **cache key**: the hash xorq uses to find a cached result. It covers what
  produced the result, so the same pipeline finds the same entry.
- **pin**: make a served expression read its cached files directly instead
  of recomputing keys (see "Serving over Flight").
- **exchange**: one Arrow Flight request: the client streams rows in and
  gets rows back.

## The token

In short: tabpfn-client finds the token on its own, can fall back to a
cached login, and re-reads the environment on every call until one succeeds.

Lookup order (`tabpfn_client/service_wrapper.py`,
`resolve_token_with_source`):

1. a token set with `set_access_token()`;
2. `TABPFN_TOKEN`;
3. a token cached by an earlier `interactive_login()`.

So a machine with a cached login still authenticates with `TABPFN_TOKEN`
unset. That cache file is `tabpfn_client/.tabpfn/config` under the venv's
`site-packages` (`tabpfn_client/constants.py`, `CACHE_DIR`). It belongs to
the venv, not the host.

A rejected `TABPFN_TOKEN` is sent again on every `fit` and `predict`. Three
things combine:

- `resolve_token_with_source` reads `os.environ` on each call;
- `_discard_token` (same module) only clears an options field that nothing
  reads back;
- `tabpfn_client/config.py` sets `Config.is_initialized` only after a
  successful `init()`, so every later call runs `init()` again.

Two consequences: exporting a new token in the same process takes effect
without a restart, and a loop with a bad token keeps hitting the auth
endpoint.

## What shapes the chain in xorq 0.4.6

In short: three workarounds in `expr.py` exist because of how xorq 0.4.6
handles estimator params, UDF names and cache keys.

- **Params must be hashable.** xorq hashes estimator params, so a
  dict-valued param fails with `TypeError: unhashable type: 'dict'`. The
  regressors' `inference_config` is therefore an ibis `FrozenDict`, a dict
  subclass the client sends unchanged.
- **Each predict needs its own `name_infix`.** xorq names a predict UDF from
  the estimator class and params alone. The severity and calibration
  regressors have the same class and params, so they would get the same
  name, and DataFusion would bind the calibration call to the severity
  call's signature (`Failed to coerce arguments`).
- **The model state is taken from inside `predicted`** (`model_state_of`),
  not rebuilt; see "Caching the fitted model" for why.

## Rate limit and record batches

In short: `whole_batches` sorts and caches the slices that get predicted, so
the default run makes 10 predicts on any machine instead of hundreds.

DataFusion splits work into one partition per CPU, so after the join and
split a slice arrives in batches of a few hundred rows on a many-core
machine. One predict per batch would mean hundreds of predicts and a blown
rate limit.

`whole_batches` fixes this in two steps, and needs both:

1. sorting by policy id merges the slice into one stream of 8192-row
   batches;
2. caching it stores those batches as parquet row groups, so a predict reads
   them back whole.

Sorting alone or caching alone leaves the batches fragmented.
`tests/test_expr.py`, `test_whole_batches_keep_predicts_to_one_per_slice`,
checks the combination at 48 partitions. At the default sample the chain
makes 10 predicts: two batches for each stage-one model on the calibration
slice, and two for each of the three models on the test slice.

This is done in the expression rather than with a session setting because
`xorq build` records the connection without its config (`config: ~`), and
`xorq run` has no option to set one.

## What TabPFN returns from fit

In short: a fitted TabPFN model is an ID pointing at data on Prior Labs'
servers, not a set of weights. If the server loses that data, the model is
gone and the only fix is to fit again.

`fit()` uploads the training set, the server fits, and the client keeps the
server-assigned UUID in `model_id_`. Every `predict()` sends that ID plus the
hyperparameters. So the fitted estimator holds only `model_id_`, `classes_`,
`_n_train_rows` and `get_params()`, and its pickle is under 10 KB
(`tests/test_expr.py`).

`save_model()` turns that into a small JSON dict (without `client_options`),
and `TabPFNClassifier.load_model(dict)` rebuilds an estimator ready to
predict without calling `fit()`. Because `predict()` authenticates on its
own, a loaded estimator works. The ID works only for the account that
created it, and only while the uploaded training data is still on the server.
Source: `tabpfn_client/persistence.py`.

What this means for the cache: a cache hit on a model whose server-side data
is gone raises `FittedModelNotFoundError` at predict time
(`tabpfn_client/client.py`, `predict`). Retrying won't help; the model must
be fit again. `scripts/evict_model.py` deletes every cache entry a model went
into, the `model_state` in `expr.py`:

- the three models;
- the staged calibration slice;
- the predictions.

The OpenML downloads and sorted slices stay. The next run then fits again.

Finding those entries means computing their cache keys, and that sends one
HEAD request to OpenML per source (see "Caching the fitted model"). To avoid
that, `serve_flight.py` saves the keys it pinned to a file in the cache
(`MODEL_STATE_RECORD`), and `evict_model.py` deletes those by default:
offline, and correct even if OpenML's files have changed since. `--from-expr`
computes the keys instead, for a cache no server has run on. After an
eviction, restart any running server (see "Serving over Flight").

## Caching the fitted model

In short: one cache holds the downloads, the slices, the models and the
predictions, so a second run makes no calls to Prior Labs. The cache keys
don't record the Prior Labs account.

`expr.py` uses one `ParquetCache`, under `XORQ_CACHE_DIR`, for:

- the two OpenML sources;
- the sorted calibration and test slices (`whole_batches`);
- the three fitted models, each the UUID and its few fields, pickled;
- the staged calibration slice;
- `predicted`.

Every later process running the same expression reads them all back
(`smoke_cache.py` checks this). `base_path` is left unset so that
`xorq run --cache-dir` can move the cache.

The OpenML sources are `deferred_read_parquet` reads with their schemas
given up front, so building the expression downloads nothing. The cache key
of a download is the URL plus the `Last-Modified` and `Content-Length`
headers from a HEAD request, so a warm cache costs one HEAD and no download.

What the keys cover:

- a model's key: its training data expression and the estimator params;
- a prediction's key: that, plus the expression it predicts on;
- none of them: the Prior Labs account the model belongs to.

Every key, including the one with the `FrozenDict` param, is the same in two
processes. Two more things about keys:

- **Don't use `deferred_model` directly.** The fit library's `deferred_model`
  property builds a new UDAF (an aggregate UDF) on every access. Its class
  name carries a counter that ends up in a column name, and so in the key.
  Taking `deferred_model` from the tuple the library returns gives a
  different key from the one the predict UDF uses, and the process fits
  twice. `expr.py` therefore takes the models from the cached nodes inside
  `predicted` (`model_state_of`).
- **Builds share the cache.** A build computes the same keys as the
  in-process expression (checked under the default modification-time
  strategy). So `xorq run` on a build reads what a `smoke_live.py` run
  cached, and vice versa.

A cached table can't be served directly. `flight_serve` calls `unbind()`,
which treats a cached table's cache node as the input, so the server would
expect the cached table's schema and the client's schema check fails. What
is served instead is `scored`, which has no cache node of its own; see
"Serving over Flight".

## What predict uploads and what the client caches

In short: predicting uploads the rows being predicted to Prior Labs, and
they stay there, just like training data. The client's size limits can be
up to an hour stale.

`predict()` (`tabpfn_client/client.py`) posts a hash of the rows to the
prepare-test-set-upload endpoint. If the server answers 409, it already has
that data; otherwise the client streams the parquet bytes to signed URLs
(`_upload_to_gcs`). The upload is stored by content and kept server-side
alongside the training set, so predicted rows raise the same data-residency
question as training data.

Before uploading, the client enforces limits on rows, columns, cells and the
number of test-by-train row pairs (`tabpfn_client/estimator.py`,
`validate_train_set` and `validate_test_set`). It fetches those limits from
the server (`tabpfn_client/client.py`, `get_settings`) and caches them in
the process for an hour, so a long-running process can use limits up to an
hour old. Fits on a training set above a server-set size (the client
defaults to 50 MiB) are submitted as a background job and polled
(`submit_fit_job`).

## Serving over Flight

In short: the server fits once at startup and then only predicts. Clients
send the nine policy columns and nothing they send can reach a fit. Each
exchange costs at least three predicts, so a server handles about 20
exchanges a minute.

### What is served

`scripts/serve_flight.py` serves `scored` as an Arrow Flight exchange, via
`serve_scored` in `expr.py`. That is xorq's
`xorq.expr.relations.flight_serve_unbound` plus an exchanger that merges the
client's batches (below).

`scored` is the chain applied to a placeholder table with only the nine
policy-feature columns, so those nine columns are the only input a client
can send. Exposure is fixed at a year inside the chain. The three fits stay
tied to their own slices and run in the server process, once per cache
directory.

It must be `flight_serve_unbound`, not `flight_serve`. `flight_serve` turns
every table in the expression into an input, the training data included, so
"send rows to score" would become "send a training set and refit".

### Pinning

At startup the server pins `scored` (`ls.pin(ensure_materialized=True)`). On
a cold cache that is where the fits happen. After that, every cached entry
is a plain read of its file. So an exchange only predicts: it computes no
cache keys and never sends OpenML a HEAD request. `tests/test_expr.py`
checks this by serving a pinned expression with its source files deleted.

A pinned read has no fallback. After an eviction, the server's exchanges
fail until it is restarted, which pins and fits again. Nothing stops two
servers sharing one cache directory from both fitting, so each would fit
each model once.

### Batching and the rate limit

A client can stream its rows in any batch sizes, and each batch would be one
predict per model: 100 small batches would be 300 predicts.
`MergingExchanger` first re-cuts the input into 8192-row batches, so an
exchange costs three predicts per 8192 rows however the client batched it,
and at least three. At Prior Labs' 60 a minute, that is about 20 exchanges
a minute, of up to 8192 rows each, across all the server's clients.

### Tests and scripts

- `scripts/flight_client.py` sends the policy features of part of the test
  split. It checks that the rows come back as sent, in order and each priced,
  and refuses to run with `TABPFN_TOKEN` set.
- `scripts/smoke_flight.py` runs one server (with the token) and two clients
  (without) on separate slices. It checks the server's calls to Prior Labs
  were three fits and ten predicts. It then starts a second server without
  the token on the warm cache, which still starts, and checks that a
  client's exchange fails at predict.
- `tests/test_expr.py` covers the same shape offline with a stand-in
  estimator:
  - the input is the policy features;
  - two exchanges cause three fits;
  - a batch carrying `ClaimAmount` is refused;
  - three simultaneous exchanges on a cold cache fit each model once;
  - a pinned server works;
  - twelve 5-row client batches cost three predicts.

### How an exchange fails

- **Extra columns.** A client batch carrying extra columns (such as
  `ClaimAmount`) is refused before the exchange opens. The client checks
  the batch schema against the server's required input
  (`xorq/flight/client.py`, `do_exchange_batches`). That check runs on the
  client, so it doesn't stop a hostile client. The real guarantee is that
  the fits are tied to their slices and never read what a client sends.
- **No token on the server.** The server logs
  `RuntimeError: No TabPFN access token found.` from
  `tabpfn_client/config.py`, `init`. `init` never prompts, and first checks
  that Prior Labs is reachable.
- **What the client sees.** The client never gets that `RuntimeError`, or
  any other server-side error such as `FittedModelNotFoundError` for a stale
  `model_id_`. It gets a gRPC error instead:
  `resource exhausted ... received initial metadata size exceeds limit`.
  pyarrow's Flight server turns a Python exception into a status containing
  the message plus the full server traceback. The failed exchange comes back
  in a form the client reads as initial metadata, and gRPC's default size
  limit for metadata throws the whole status away. Tracked upstream as
  xorq-labs/xorq#2269.

## Where code must live

In short: keep any code the models or UDFs need in `expr.py`. Code imported
from `src/` breaks when someone runs a plain `builds/` directory.

`xorq build` pickles everything the expression references:

- functions and classes defined in `expr.py` by value, so the code travels
  inside the build (the script is loaded as `__main__`);
- anything imported from `src/xorq_template_tabpfn/` by reference, so the
  package must be installed wherever the build runs.

Whether that reference resolves depends on how the build is run.
`scripts/src_import_expr.py` is an offline expression whose UDF imports a
function from `src/`, used to check each case:

| run in a venv that has only xorq 0.4.6 | `src/` import |
| --- | --- |
| the `builds/<hash>` directory (`xorq run`) | fails, `ModuleNotFoundError`: `xorq run` installs neither the wheel nor `requirements.txt`, even when `xorq uv build` or `xorq catalog add` has put them in the build dir |
| the catalog entry's archive, unpacked, after `pip install -r requirements.txt` and the wheel it carries (`xorq run`) | works: `xorq catalog add` builds the wheel and exports the lock into the archive (`xorq/catalog/catalog.py`, `_ensure_wheel_artifacts`) |
| the catalog entry through `xorq catalog run` | works: the default runner is `uv tool run xorq run` with the entry's wheel and `requirements.txt` (`xorq/ibis_yaml/packager.py`, `PackagedRunner`) |

Today everything the models and UDFs need is in `expr.py`. A `src/` import
is safe only if everyone runs the catalog entry and nobody runs a plain
build directory.
