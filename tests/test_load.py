"""Load test notebook against a simulated endpoint, with a real MLflow tracking store (SQLite): above 3 requests in
flight the assistant's retrieval step silently returns nothing (HTTP 200, no citation), above 8 the endpoint answers
HTTP 429. Checks that both are detected from the right level and that every request is an MLflow trace: 429 traces in
state ERROR with the HTTP error recorded as an exception, the assistant's steps copied, the empty retrieval step
marked ERROR, tags, the "outcome" assessment, and the link to the level run."""
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import types
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
from harness import Widgets, run_cells  # noqa: E402

import mlflow  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
work = tempfile.mkdtemp()
mlflow.set_tracking_uri(f"sqlite:///{work}/mlflow.db")
in_flight, lock = [0], threading.Lock()


class Response:
    def __init__(self, status, data=None, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}
        self.ok = status == 200
        self.text = "" if self.ok else '{"code":"429","message":"Rate limit exceeded. Please try again later."}'

    def json(self):
        return self._data


def assistant_trace(docs: bool, t0: int) -> dict:
    """Trace returned with databricks_options.return_trace (serialized MLflow spans, server clock)."""
    def span(sid, parent, name, kind, start, end, outputs=None):
        return {"span_id": sid, "parent_span_id": parent, "name": name, "start_time_unix_nano": t0 + start,
                "end_time_unix_nano": t0 + end, "status": {"code": "STATUS_CODE_OK", "message": ""}, "events": [],
                "attributes": {"mlflow.spanType": json.dumps(kind), "mlflow.spanOutputs": json.dumps(outputs),
                               "mlflow.databricksHideRetriever": "true"}}
    passages = [{"page_content": "[Source: QP-1457 | Title: Records] Records are kept 10 years.",
                 "metadata": {"doc_uri": "QP-1457"}}] if docs else []
    spans = [span("a", None, "Knowledge Assistant", "AGENT", 0, 40_000_000),
             span("b", "a", "docs", "RETRIEVER", 1_000_000, 2_000_000, passages),
             span("c", "a", "Final_response", "CHAIN", 5_000_000, 39_000_000, "answer")]
    if docs:
        spans.append(span("d", "a", "rerank", "RERANKER", 2_000_000, 4_000_000, passages))
    return {"info": {"trace_id": f"tr-{time.time_ns():032x}"}, "data": {"spans": spans}}


def post(url, json=None, headers=None, timeout=None):
    with lock:
        in_flight[0] += 1
        load = in_flight[0]
    time.sleep(0.05)
    try:
        if load > 8:
            return Response(429)
        docs = load <= 3
        text = "Selon **QP-1457**, les enregistrements sont conservés 10 ans." if docs else \
            "Je n'ai pas trouvé cette information dans la documentation."
        ann = [{"type": "url_citation", "url": "https://x/identification.aspx?ref=QP-1457#:~:text=a"}] if docs else []
        return Response(200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": text,
                                                                          "annotations": ann}]}],
                              "custom_outputs": {"sources_used": docs},
                              "databricks_output": {"trace": assistant_trace(docs, time.time_ns() + 7 * 10**9)}})
    finally:
        with lock:
            in_flight[0] -= 1


import requests  # noqa: E402

requests.post = post
cfg = MagicMock()
cfg.host, cfg.authenticate.return_value = "https://example", {"Authorization": "Bearer x"}
import databricks.sdk  # noqa: E402
import databricks.sdk.core  # noqa: E402

databricks.sdk.core.Config = lambda: cfg
databricks.sdk.WorkspaceClient = MagicMock

shown = []
ns = {"dbutils": types.SimpleNamespace(widgets=Widgets({
          "concurrency_levels": "2,5,20", "requests_per_level": "40", "n_questions": "3",
          "pause_between_levels_s": "0", "write_tables": "false", "experiment_path": "/Shared/load-test"})),
      "spark": MagicMock(), "display": lambda d, *a, **k: shown.append(d)}
run_cells(str(REPO / "Load_Test_Knowledge_Assistant.py"), ns, skip=("Setup",))

s = ns["summary"].set_index("phase")
print(s[["concurrency", "n", "ok", "http_429", "retriever_empty", "latency_p50_s", "throughput_rps"]].to_string())
assert s.loc["baseline", ["http_429", "retriever_empty"]].sum() == 0
assert s.loc["load_2", ["http_429", "retriever_empty"]].sum() == 0
assert s.loc["load_5", "retriever_empty"] > 0 and s.loc["load_5", "http_429"] == 0
assert s.loc["load_20", "http_429"] > 0
print("✓ 429 and silent retrieval failures detected at the expected levels")

df = ns["df"]
traces = {t.info.trace_id: t for t in mlflow.search_traces(locations=[ns["EXPERIMENT_ID"]], return_type="list",
                                                            max_results=1000)}
assert len(traces) == len(df), (len(traces), len(df))
throttled = traces[df[df["outcome"] == "http_429"]["trace_id"].iloc[0]]
empty = traces[df[df["outcome"] == "retriever_empty"]["trace_id"].iloc[0]]
fine = traces[df[(df["outcome"] == "ok") & (df["phase"] == "load_2")]["trace_id"].iloc[0]]

assert str(throttled.info.state).endswith("ERROR"), throttled.info.state
call = next(sp for sp in throttled.data.spans if sp.name.startswith("POST "))
assert call.status.status_code.name == "ERROR" and any(e.name == "exception" for e in call.events)
assert throttled.info.tags["outcome"] == "http_429" and throttled.info.tags["status_code"] == "429"

names = {sp.name: sp for sp in empty.data.spans}
assert {"load_request", "Knowledge Assistant", "docs", "Final_response"} <= set(names), set(names)
assert names["docs"].status.status_code.name == "ERROR" and "0 documents" in names["docs"].status.description
assert str(empty.info.state).endswith("ERROR")
call = next(sp for sp in empty.data.spans if sp.name.startswith("POST "))
for sp in empty.data.spans:
    if sp.name in ("Knowledge Assistant", "docs", "Final_response"):
        assert call.start_time_ns <= sp.start_time_ns and sp.end_time_ns <= call.end_time_ns, sp.name

assert str(fine.info.state).endswith("OK")
docs = next(sp for sp in fine.data.spans if sp.name == "docs")
assert docs.status.status_code.name == "OK" and len(docs.outputs) == 1
assert fine.info.tags["assistant_trace_id"].startswith("tr-")
assert any(a.name == "outcome" and a.value == "ok" for a in fine.info.assessments)
level_run = ns["LEVEL_RUNS"]["load_2"]
assert fine.info.request_metadata.get("mlflow.sourceRun") == level_run
print("✓ traces: 429 in ERROR with the exception, assistant steps copied inside the call, empty retrieval marked ERROR,"
      " tags, outcome assessment, linked to the level run")

client = mlflow.MlflowClient()
artifacts = {a.path for a in client.list_artifacts(ns["PARENT_RUN_ID"], "charts")}
assert {"charts/outcomes_by_level.png", "charts/request_timeline.png", "charts/latency_throughput.png"} <= artifacts
history = client.get_metric_history(ns["PARENT_RUN_ID"], "http_429_rate")
assert [m.step for m in history] == [1, 2, 5, 20] and history[-1].value > 0
print("✓ parent run: charts and per-level metrics with the concurrency as step")
