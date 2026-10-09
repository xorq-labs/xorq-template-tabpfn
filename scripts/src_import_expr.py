"""An expression whose UDF imports code from src/, for checking where it resolves.

The UDF below calls ``label`` imported from the installed package, so
cloudpickle stores a reference to ``xorq_template_tabpfn.labels`` rather
than the function's bytes. Running its build, or its catalog entry, with
``xorq run`` in a venv that has only xorq shows whether that reference
resolves; docs/internals.md, "Where code must live", states which artifacts
resolve it. Offline: no TabPFN, no token.
"""

from __future__ import annotations

import pandas as pd

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.expr.udf import make_pandas_udf
from xorq_template_tabpfn.labels import label


rows = xo.memtable(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

labelled = make_pandas_udf(
    fn=lambda df: df["x"].map(label),
    schema=rows.schema(),
    return_type=dt.string,
    name="label_from_src",
)

expr = rows.mutate(labelled.on_expr(rows).name("label"))
