"""The httpx request-line counter the live smoke scripts share."""

from __future__ import annotations

import logging
import re


# tabpfn-client posts a fit to one of two endpoints depending on its mode and
# the training-set size; both count as a fit.
FIT_ENDPOINTS = ("fit", "submit_fit_job")
# expr.py's three models; smoke_cache.py compares their ids across processes.
MODEL_NAMES = ("claim_model", "severity_model", "calibration_model")


class PostCounter(logging.Handler):
    """Counts httpx's INFO-level request lines by ``/tabpfn/`` endpoint."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.posts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        self.count(record)

    def count(self, record: logging.LogRecord) -> str | None:
        """Count the record's POST and return its endpoint; None for any other line."""
        if m := re.search(r'POST \S*/tabpfn/(\w+) "', record.getMessage()):
            endpoint = m.group(1)
            self.posts[endpoint] = self.posts.get(endpoint, 0) + 1
            return endpoint
        return None

    def attach(self) -> PostCounter:
        """Hook into the httpx logger, after tabpfn_client has been imported.

        tabpfn_client lowers that logger to WARNING at import, so the level
        has to be raised back afterwards for the request lines to exist.
        """
        logger = logging.getLogger("httpx")
        logger.setLevel(logging.INFO)
        logger.addHandler(self)
        return self

    def since(self, before: dict[str, int]) -> dict[str, int]:
        """POSTs per endpoint since ``before`` (a copy of ``posts``), nonzero only.

        Every endpoint that was hit is reported, so a POST to an endpoint the
        caller did not expect shows up rather than vanishing.
        """
        deltas = {
            endpoint: self.posts.get(endpoint, 0) - before.get(endpoint, 0)
            for endpoint in sorted(self.posts.keys() | before.keys())
        }
        return {endpoint: n for endpoint, n in deltas.items() if n}
