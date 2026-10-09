# xorq-template-tabpfn

[TabPFN](https://docs.priorlabs.ai) is a Prior Labs model for tabular data
that runs on their servers: every fit and every prediction is an API call on
your account, and Prior Labs allows 60 predictions a minute. This template
runs TabPFN through [xorq](https://github.com/xorq-labs/xorq), which describes
the whole pipeline up front and caches every step. The result:

- each model is fit **once**; later runs read it back from the cache and make
  no calls to Prior Labs;
- predictions can be served over the network (Arrow Flight) to clients that
  have no Prior Labs token.

The worked example in `expr.py` prices French car-insurance policies, following
Prior Labs' "Insurance Claim Modeling" cookbook
(docs.priorlabs.ai/cookbook/insurance_claim_modeling). One model predicts
*whether* a policy will have a claim, a second predicts *how much* a claim
costs, and a third corrects their combined estimate. "How the example works"
below walks through it; "Adapting the template" says what to change for your
own data.

Your token stays in the environment of the process that does the fits.
tabpfn-client reads `TABPFN_TOKEN` itself at call time, so the token never
enters a build, the cache, or a client. How the lookup works is in
docs/internals.md, "The token".

## Quickstart

You need:

- [direnv](https://direnv.net), which loads the environment when you `cd` in
- [uv](https://docs.astral.sh/uv/), which installs the Python dependencies
- [sops](https://github.com/getsops/sops) and an encryption key of your own
  (for example an [age](https://github.com/FiloSottile/age) key)
- a Prior Labs account and an API token from
  https://platform.priorlabs.ai/account/api-keys

### 1. Store your token, encrypted

The token is personal: tokens and fitted models belong to one Prior Labs
account, so each developer encrypts their own, and the file is gitignored.
Write it to `.envrcs/.env.priorlabs.sops-encrypted` as an encrypted dotenv
file, for example with an age key:

```bash
printf 'TABPFN_TOKEN=%s\n' '<your-token>' > token.env
sops encrypt --age <your-age-public-key> --input-type dotenv --output-type dotenv \
  token.env > .envrcs/.env.priorlabs.sops-encrypted
rm token.env
```

### 2. Load the environment

```bash
direnv allow
```

This creates `.envrcs/.envrc.secrets` and `.envrcs/.envrc.user` from their
templates, decrypts your token into the environment, and installs the
dependencies with uv. The `.envrcs/` layout comes from
`xorq-labs/xorq-envrcs`; its `.envrcs/README.md` is the reference.

### 3. Check it offline

```bash
python -m pytest      # no calls to Prior Labs or OpenML; the tests block the network
xorq build expr.py    # writes builds/<hash>; makes no calls
pre-commit install    # once per clone: runs CI's lint checks on every commit
```

### 4. Run it live

These scripts call Prior Labs on **your** account. Run one when a change
affects what it checks, not in a loop.

```bash
python scripts/smoke_live.py    # fits the three models, predicts, prints the metrics
python scripts/smoke_cache.py   # runs smoke_live.py twice; the second run must make no calls
python scripts/smoke_flight.py  # starts a prediction server and two token-less clients
```

## Layout

| path | what it is |
| --- | --- |
| `expr.py` | the whole pipeline. Its two outputs are `predicted`, the test set scored, and `scored`, the same chain applied to rows a client sends |
| `scripts/smoke_*.py` | the live checks above |
| `scripts/serve_flight.py`, `scripts/flight_client.py` | a prediction server and a token-less client for it |
| `scripts/evict_model.py` | deletes cached models that Prior Labs no longer recognizes (see "Caching") |
| `scripts/src_import_expr.py`, `src/xorq_template_tabpfn/` | an example of importing code from `src/`; see "Adapting the template" |
| `tests/test_expr.py` | offline tests, using stand-in models and made-up policies |

## How the example works

The goal is a **pure premium**: the expected claim cost of a policy over one
year. The data is freMTPL2, about 678,000 French car-insurance policies.
Each policy has an **exposure**, the fraction of a year it was in force, so
predictions are made at a full year of exposure to give a yearly figure.

1. **Load and join** (`read_openml`, `make_policies`). Two OpenML tables are
   read: one row per policy (dataset 41214) and one row per claim (41215).
   Claims are summed per policy and joined on. As in the cookbook, exposure
   is capped at one year and claim amounts at 200,000.
2. **Split** (`make_slices`). A random sample of the policies is divided into
   three non-overlapping parts: a training set, a calibration set and a test
   set. `SAMPLE_FRACTION` (default 0.075) sets the sample size. The default
   gives about 30k / 10k / 10k policies.
3. **Batch** (`whole_batches`). The calibration and test sets are sorted and
   cached so they are read in large chunks. Every chunk costs one call to
   Prior Labs, so this keeps the run under the rate limit. Details are in
   docs/internals.md, "Rate limit and record batches".
4. **Predict** (`make_scorer`). Two models are trained on the training set:
   - a `TabPFNClassifier` for the probability that a policy has a claim;
   - a `TabPFNRegressor`, trained only on policies with a claim, for the
     claim amount.

   Their product is `premium`.
5. **Calibrate.** A second `TabPFNRegressor` is trained on the calibration
   set. Its inputs are the two predictions from step 4 plus the policy's
   actual exposure, and its output, applied to the test set, is
   `premium_calibrated`.

Why calibration needs its own data: TabPFN doesn't learn weights. It reads
its training rows at prediction time. So a prediction for a row it trained
on is unrealistically accurate, like grading an exam with the answer key in
hand, and calibrating against such predictions would teach the wrong
correction. The cookbook makes the same point about its fixed correction
factor, which this model replaces.

`smoke_live.py` prints the cookbook's two metrics for `premium` and
`premium_calibrated`:

- **Gini** (weighted by exposure): how well the premiums rank policies from
  riskiest to safest. Higher is better.
- **Balance**: total premium charged divided by total claims paid. 1.0 means
  the premiums cover the claims exactly.

The cookbook trains on 508,509 policies, not 30k, so its numbers are not
comparable to the default run's.

## Adapting the template

The example's specifics all live in `expr.py`:

- **Data:** replace `read_openml` and `make_policies` (and the
  `FREQ_*`/`SEV_*` constants) with your own sources, producing one row per
  record.
- **Columns:** set `POLICY_FEATURES`, `FEATURES` and the targets
  (`CLAIM_TARGET`, `AMOUNT_TARGET`) to your columns.
- **Models:** `make_scorer` wires the three models together. A single-model
  pipeline needs only one `deferred_fit_predict_sklearn` call. Keep the
  estimator classes and params as arguments so the tests can substitute
  stand-ins.
- **Sample size:** `SAMPLE_FRACTION` and `SLICE_SHARES`.

Keep any function the models or predictions need in `expr.py`. Code
imported from `src/` works only when the build is run from the catalog, not
from a plain `builds/` directory (docs/internals.md, "Where code must live").

## Caching

Downloaded data, the sorted calibration and test sets, the fitted models and
the predictions are all cached under `XORQ_CACHE_DIR`. A second run of the
same pipeline reads everything back and makes no calls to Prior Labs
(`smoke_cache.py` checks this). To move the cache, use
`xorq run --cache-dir`.

A cached model is a reference to a model stored on Prior Labs' servers under
the account that fit it, not a copy. It breaks if Prior Labs deletes the
stored model, or if you run with a different account's token. Either way,
predictions fail with `FittedModelNotFoundError`. To fix it, delete the
cached models and the predictions made with them (the downloaded data
stays), then run again to refit:

```bash
python scripts/evict_model.py              # if a Flight server has run on this cache
python scripts/evict_model.py --from-expr  # otherwise (e.g. after smoke_live.py)
```

Then restart any running Flight server. What a cached model is, and why the
two forms differ, is in docs/internals.md, "What TabPFN returns from fit";
what the cache holds and what its keys cover is in "Caching the fitted
model".

## Serving over Flight

`scripts/serve_flight.py` runs a server that prices policies. A client sends
a table with the nine columns in `POLICY_FEATURES` and gets the same rows
back with the predictions added, `premium_calibrated` last. The server holds
`TABPFN_TOKEN` and fits the models when it starts, once per cache directory.
Clients need no token, and `flight_client.py` refuses to run if one is set.

```bash
python scripts/serve_flight.py --port 8815
env -u TABPFN_TOKEN python scripts/flight_client.py --port 8815 --offset 0 --limit 40
```

> **Mind the rate limit.** Each request costs three calls to Prior Labs per
> 8,192 rows, whatever batch sizes the client uses, and Prior Labs allows 60
> calls a minute. Send policies in a few large requests, not one per policy.

How the server is built, why batching works this way, and how a request
fails are in docs/internals.md, "Serving over Flight".

## Builds and the catalog

`xorq build expr.py` writes a self-contained snapshot of the pipeline to
`builds/<hash>`. Build from a clean, committed tree: uncommitted edits are
recorded in the build's `build_metadata.json` and travel with it into the
catalog. Running a build needs `TABPFN_TOKEN` in the environment; the build
itself never contains it.

`xorq uv build expr.py` also packages the dependencies. For that, the build's
`requirements.txt` must match the repo's byte for byte (see "Updating
dependencies").

A **catalog** is a git repository of published builds that others can fetch
and run. This repo's builds are published to
`github.com/xorq-labs/xorq-catalog-template-tabpfn` under the name
`tabpfn-insurance`.

```bash
xorq catalog clone https://github.com/xorq-labs/xorq-catalog-template-tabpfn.git --name xorq-catalog-template-tabpfn
xorq catalog --name xorq-catalog-template-tabpfn get <entry-hash> -o <dir>   # writes <dir>/<entry-hash>.zip; <dir> must exist
```

`get` needs the entry's hash, not the name `tabpfn-insurance`. In xorq 0.4.6
the CLI can't look up names here, though the Python API's `Catalog.load` can.

To add a build to your own catalog:

```bash
xorq catalog --name <your-catalog> add --no-sync builds/<hash>
```

> **Warning:** without `--no-sync`, `catalog add` pushes to the catalog's
> remote.

## Updating dependencies

xorq is pinned to a PyPI release (`xorq[duckdb]==0.4.6`). To change it, edit
the pin in `pyproject.toml`, then:

```bash
uv lock
uv export --locked --no-dev --no-emit-project --no-header --no-annotate > requirements.txt
```

Use exactly these export flags. `xorq uv build` compares its own export with
the repo's `requirements.txt` byte for byte, and different flags make it fail
(see [xorq-labs/xorq#1941](https://github.com/xorq-labs/xorq/issues/1941)).
Pin xorq to a git source instead only on a branch that needs an unreleased
xorq change.
