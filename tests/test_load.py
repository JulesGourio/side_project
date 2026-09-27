"""Load test notebook against a simulated endpoint: above 3 requests in flight the retriever silently returns nothing
(HTTP 200, no citation), above 8 the endpoint answers HTTP 429. Checks that both are detected, from the right level."""
import os
import pathlib
import sys
import threading
import time
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from harness import Widgets, run_cells  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
in_flight, lock = [0], threading.Lock()


class Response:
    def __init__(self, status, data=None, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}
        self.ok, self.text = status == 200, "" if status == 200 else "REQUEST_LIMIT_EXCEEDED"

    def json(self):
        return self._data


def post(url, json=None, headers=None, timeout=None):
    with lock:
        in_flight[0] += 1
        load = in_flight[0]
    time.sleep(0.05)
    try:
        if load > 8:
            return Response(429, headers={"Retry-After": "1"})
        docs = load <= 3
        text = "Selon **QP-1457**, les enregistrements sont conservés 10 ans." if docs else \
            "Je n'ai pas trouvé cette information dans la documentation."
        ann = [{"type": "url_citation", "url": "https://x/identification.aspx?ref=QP-1457#:~:text=a"}] if docs else []
        spans = [{"name": "retrieve", "attributes": {"mlflow.spanType": "\"RETRIEVER\"",
                                                     "mlflow.spanOutputs": '[{"page_content": "x"}]' if docs else "[]"}}]
        return Response(200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": text,
                                                                          "annotations": ann}]}],
                              "custom_outputs": {"sources_used": docs},
                              "databricks_output": {"trace": {"data": {"spans": spans}}}})
    finally:
        with lock:
            in_flight[0] -= 1


import requests  # noqa: E402

requests.post = post
cfg = MagicMock()
cfg.host, cfg.authenticate.return_value = "https://example", {"Authorization": "Bearer x"}
import databricks.sdk.core  # noqa: E402

databricks.sdk.core.Config = lambda: cfg

shown = []
ns = {"dbutils": types.SimpleNamespace(widgets=Widgets({"concurrency_levels": "2,5,20", "requests_per_level": "40",
                                                        "pause_between_levels_s": "0", "write_results": "false"})),
      "spark": MagicMock(), "display": lambda d, *a, **k: shown.append(d)}
run_cells(str(REPO / "Load_Test_Knowledge_Assistant.py"), ns)
print(shown[0][["phase", "concurrency", "n", "http_429", "retriever_empty", "lost_documents", "no_documents",
                "not_found_answers"]].to_string())
s = shown[0].set_index("phase")
assert s.loc["baseline", ["http_429", "retriever_empty"]].sum() == 0
assert s.loc["load_2", ["http_429", "retriever_empty"]].sum() == 0
assert s.loc["load_5", "retriever_empty"] > 0 and s.loc["load_5", "http_429"] == 0
assert s.loc["load_20", "http_429"] > 0
print("✓ 429 and silent retrieval failures detected at the expected levels")
