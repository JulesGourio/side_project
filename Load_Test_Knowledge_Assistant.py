# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Load Test
# MAGIC
# MAGIC Sends concurrent requests to a Knowledge Assistant serving endpoint, one concurrency level after the other, and
# MAGIC records **every request as an MLflow trace**, including the ones the endpoint rejects. It detects the two failure
# MAGIC modes seen under load:
# MAGIC - **rate limiting**: HTTP 429 (and 5xx, timeouts), with the `Retry-After` header when present;
# MAGIC - **silent retrieval failures**: HTTP 200 and a well-formed answer, but the retrieval step returned nothing (or
# MAGIC   raised an error it swallowed), so the answer cites no document.
# MAGIC
# MAGIC ### Why the assistant's own experiment shows neither
# MAGIC - A request rejected with HTTP 429 is refused by the serving layer **before the assistant runs**: no trace is
# MAGIC   written in the assistant's experiment. Only the caller sees it, so this notebook traces from the caller's side.
# MAGIC - A silent retrieval failure is logged by the assistant with every step `OK`: an empty `RETRIEVER` step and no
# MAGIC   `RERANKER` step. This notebook copies the assistant's steps into its own trace and marks that step `ERROR`.
# MAGIC
# MAGIC ### What the MLflow experiment shows
# MAGIC | Where | What |
# MAGIC |---|---|
# MAGIC | **Runs** | one parent run per test (parameters, metrics per level with the concurrency as step, charts, request table) and one child run per level |
# MAGIC | **Traces** | one trace per request, linked to its level run. State `ERROR` for every anomaly; tags `outcome`, `status_code`, `phase`, `concurrency`, `attempts`, `retriever_docs`, `citations`, `assistant_trace_id`; assessment `outcome` |
# MAGIC | Trace steps | `load_request` → `POST <endpoint>` per attempt (HTTP status, latency; an HTTP error is recorded as an exception) → `backoff` between attempts → the assistant's own steps copied from the trace it returns, retrieved documents included |
# MAGIC
# MAGIC Filter the Traces tab with `tags.outcome = 'http_429'` or `tags.outcome = 'retriever_empty'`, or sort by State.
# MAGIC
# MAGIC ### Outcomes
# MAGIC | Outcome | Meaning |
# MAGIC |---|---|
# MAGIC | `ok` | HTTP 200, documents retrieved and cited |
# MAGIC | `http_429` | rejected by rate limiting |
# MAGIC | `http_5xx`, `timeout`, `http_error`, `transport_error` | other failures of the call |
# MAGIC | `step_error` | a step of the assistant raised or logged an error (e.g. its Vector Search or reranker call failed) |
# MAGIC | `retriever_empty` | the retrieval step returned no document |
# MAGIC | `lost_documents` | no document cited, whereas the same question had documents in the baseline |
# MAGIC | `no_documents` | no document cited, and none in the baseline either (question without an answer in the documentation) |
# MAGIC | `empty_answer` | HTTP 200 without answer text |
# MAGIC
# MAGIC ### Method
# MAGIC 1. **Baseline**: every question once, one request at a time: which questions normally come back with documents.
# MAGIC 2. **Load levels** (e.g. 5, 10, 20, 40 requests in flight): the questions cycled, with that many requests in flight.
# MAGIC 3. `max_retries = 0` (default) counts every 429; a value above 0 reproduces a client that retries, and shows the
# MAGIC    attempts and the waiting time in each trace.
# MAGIC
# MAGIC ⚠️ The test loads a real endpoint: run it outside working hours on an endpoint used by people; every request is
# MAGIC billed like a normal question.

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

NEEDED = {"mlflow": (3, 11)}  # spans with explicit timestamps, trace tags from worker threads


def _as_tuple(text):
    return tuple(int(x) for x in text.split(".")[:2] if x.isdigit())


def _installed(pkg):
    for name in ([pkg, pkg + "-skinny"] if pkg == "mlflow" else [pkg]):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    return None


missing = [("mlflow[databricks]" if p == "mlflow" else p) + ">=" + ".".join(map(str, v))
           for p, v in NEEDED.items() if not _installed(p) or _as_tuple(_installed(p)) < v]
if missing:
    # Every package already present is pinned: pip either adds what is missing or fails explicitly,
    # and cannot downgrade core packages such as protobuf (which would prevent the kernel from starting).
    pins = [l for l in subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().splitlines()
            if "==" in l and not l.lower().startswith("mlflow")]
    with open("/tmp/pinned_packages.txt", "w") as f:
        f.write("\n".join(pins))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/pinned_packages.txt", *missing])
    dbutils.library.restartPython()
print({p: _installed(p) for p in NEEDED})

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("endpoint", "ka-7679a56e-endpoint")                  # serving endpoint to test (qualibot_ALL_v2)
dbutils.widgets.text("concurrency_levels", "5,10,20,40")                  # requests in flight, one level after the other
dbutils.widgets.text("requests_per_level", "40")                          # requests sent at each load level
dbutils.widgets.dropdown("questions_source", "fixed", ["fixed", "logs", "synthetic"])
dbutils.widgets.text("n_questions", "10")                                 # questions used (cycled at each level)
dbutils.widgets.text("max_retries", "0")                                  # 0 = every 429 counted; >0 = client retries
dbutils.widgets.text("timeout_s", "180")                                  # per-request timeout
dbutils.widgets.text("pause_between_levels_s", "60")                      # lets rate limits reset between levels
dbutils.widgets.text("experiment_path", "/Shared/qualibot-load-tests")
dbutils.widgets.text("assistant_experiment_id", "")                       # optional: the assistant's own experiment
dbutils.widgets.dropdown("write_tables", "true", ["true", "false"])       # Unity Catalog tables
dbutils.widgets.text("output_schema", "uat_proj.qualibot")
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")

ENDPOINT = dbutils.widgets.get("endpoint").strip()
LEVELS = sorted({int(x) for x in dbutils.widgets.get("concurrency_levels").split(",") if x.strip()})
REQUESTS_PER_LEVEL = int(dbutils.widgets.get("requests_per_level"))
QUESTIONS_SOURCE = dbutils.widgets.get("questions_source")
N_QUESTIONS = int(dbutils.widgets.get("n_questions"))
MAX_RETRIES = max(0, int(dbutils.widgets.get("max_retries") or 0))
TIMEOUT_S = float(dbutils.widgets.get("timeout_s"))
PAUSE_S = float(dbutils.widgets.get("pause_between_levels_s"))
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
ASSISTANT_EXPERIMENT_ID = dbutils.widgets.get("assistant_experiment_id").strip()
WRITE_TABLES = dbutils.widgets.get("write_tables") == "true"
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
REQUESTS_TABLE = f"{OUTPUT_SCHEMA}.ka_load_test_requests"
SUMMARY_TABLE = f"{OUTPUT_SCHEMA}.ka_load_test_summary"
SQL_WAREHOUSE_ID = "5890912c31867b77"      # reads traces stored in Unity Catalog (assistant experiment)
BACKOFF_CAP_S = 30.0

# Questions that the documentation answers (they should come back with documents)
FIXED_QUESTIONS = [
    "que signifie l'acronyme APO ?",
    "Trouve moi le template du CMP",
    "Comment suivre les compétences des opérateurs ?",
    "Quelle est la règle concernant les FAI pour des pièces qui n'ont pas été fabriquées depuis plus de 2 ans ?",
    "Quelle est la procédure de sélection des fournisseurs ?",
    "Quelle est la méthodologie d'analyse de risques des moyens de production ?",
    "Comment gérer un produit non conforme ?",
    "Quelle est la durée de conservation des enregistrements qualité ?",
    "What is the procedure for managing work centers?",
    "Qui valide un premier article (FAI) ?",
]
print(f"{ENDPOINT} · baseline then levels {LEVELS} · {REQUESTS_PER_LEVEL} requests per level · "
      f"questions: {QUESTIONS_SOURCE} · max retries: {MAX_RETRIES}")

# COMMAND ----------

# DBTITLE 1,Questions
if QUESTIONS_SOURCE == "logs":
    # A stable random sample of real user questions (the app's division prefix removed)
    QUESTIONS = [r.question for r in spark.sql(f"""
        SELECT question FROM (
            SELECT DISTINCT
                CASE WHEN content LIKE '[Division:%' AND LOCATE('state it explicitly.', content) > 0
                     THEN TRIM(SUBSTRING(content, LOCATE('state it explicitly.', content) + 20))
                     ELSE TRIM(content) END AS question
            FROM {SOURCE_SCHEMA}.chat_messages
            WHERE role = 'user' AND status = 'ok' AND deleted = false)
        WHERE LENGTH(question) BETWEEN 15 AND 400
        ORDER BY xxhash64(question)
        LIMIT {N_QUESTIONS}""").collect()]
elif QUESTIONS_SOURCE == "synthetic":
    # Synthetic retrieval questions generated from the documents, whole corpus (division ALL)
    QUESTIONS = [r.question for r in spark.sql(f"""
        SELECT DISTINCT question FROM {SOURCE_SCHEMA}.synthetic_retrieval_questions_v2
        WHERE division = 'ALL' ORDER BY xxhash64(question) LIMIT {N_QUESTIONS}""").collect()]
else:
    QUESTIONS = FIXED_QUESTIONS[:N_QUESTIONS]
print(f"{len(QUESTIONS)} questions")
for q in QUESTIONS:
    print(" -", q[:120])

# COMMAND ----------

# DBTITLE 1,Response parsing — answer, citations, and the steps of the trace returned by the assistant
import json
import re

_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def _walk(x):
    """Every dict nested in a JSON value."""
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def parse_answer(data: dict) -> dict:
    """Answer text, cited documents and sources_used flag of a Responses-format (or Chat-format) answer."""
    texts, refs, sources_used = [], [], None
    for item in (data.get("output") or []):
        for c in (item.get("content") or []) if isinstance(item, dict) else []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                texts.append(c.get("text") or "")
                for a in c.get("annotations") or []:
                    m = _REF_IN_URL.search(str(a.get("url") or a.get("title") or ""))
                    if m:
                        refs.append(m.group(1))
    for d in _walk({k: v for k, v in data.items() if k != "databricks_output"}):
        if isinstance(d.get("custom_outputs"), dict) and "sources_used" in d["custom_outputs"]:
            sources_used = bool(d["custom_outputs"]["sources_used"])
    if not texts and data.get("choices"):
        texts = [data["choices"][0]["message"].get("content") or ""]
    answer = "\n".join(texts)
    return {"answer": answer, "refs": sorted(set(refs or _REF_IN_URL.findall(answer))), "sources_used": sources_used}


def _decoded(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return value


def _status(s: dict) -> tuple:
    """(is_error, message) of a span, whatever the serialization (OpenTelemetry-like or MLflow 2)."""
    st = s.get("status")
    code, msg = (st.get("code", st.get("status_code")), st.get("message", st.get("description"))) \
        if isinstance(st, dict) else (s.get("status_code", st), s.get("status_message"))
    return ("ERROR" in str(code).upper() or code == 2), str(msg or "")


def assistant_spans(trace) -> list:
    """Steps of the trace returned with databricks_options.return_trace, normalized:
    id, parent, name, type, start/end (ns), inputs, outputs, error flag and message, events, other attributes."""
    spans = []
    for s in ((trace or {}).get("data") or {}).get("spans") or []:
        attrs = s.get("attributes") or {}
        is_error, message = _status(s)
        events = [{"name": str(e.get("name") or "event"),
                   "time": int(e.get("time_unix_nano") or e.get("timestamp") or 0),
                   "attributes": {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
                                  for k, v in (e.get("attributes") or {}).items()}}
                  for e in s.get("events") or []]
        exception = next((e["attributes"].get("exception.message") or e["attributes"].get("exception.type")
                          or "exception" for e in events if e["name"] in ("exception", "error")), None)
        spans.append({
            "id": s.get("span_id") or (s.get("context") or {}).get("span_id"),
            "parent": s.get("parent_span_id") or s.get("parent_id") or None,
            "name": s.get("name") or "step",
            "type": str(s.get("span_type") or _decoded(attrs.get("mlflow.spanType")) or "UNKNOWN").upper(),
            "start": int(s.get("start_time_unix_nano") or s.get("start_time_ns") or s.get("start_time") or 0),
            "end": int(s.get("end_time_unix_nano") or s.get("end_time_ns") or s.get("end_time") or 0),
            "inputs": s.get("inputs", _decoded(attrs.get("mlflow.spanInputs"))),
            "outputs": s.get("outputs", _decoded(attrs.get("mlflow.spanOutputs"))),
            "error": is_error or exception is not None,
            "message": message or str(exception or ""),
            "events": events,
            "attributes": {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
                           for k, v in attrs.items() if not k.startswith("mlflow.")},
            "hidden": _decoded(attrs.get("mlflow.databricksHideRetriever")) in (True, "true"),
        })
    return spans


def documents_of(outputs) -> list:
    """Documents returned by a retrieval step (a list, or the first list of a dict)."""
    if isinstance(outputs, dict):
        outputs = next((v for v in outputs.values() if isinstance(v, list)), [])
    return outputs if isinstance(outputs, list) else []


def retrieval_diagnostics(spans: list) -> dict:
    """What the assistant's trace says about its retrieval."""
    retrievers = [s for s in spans if s["type"] == "RETRIEVER"]
    errors = [f"{s['name']}: {s['message'][:200]}" for s in spans if s["error"]]
    return {
        "assistant_steps": len(spans),
        "retriever_steps": len(retrievers) if spans else None,
        "retriever_docs": sum(len(documents_of(s["outputs"])) for s in retrievers) if spans else None,
        "reranker_step": any(s["type"] == "RERANKER" for s in spans) if spans else None,
        "retriever_hidden": any(s["hidden"] for s in retrievers) if spans else None,
        "step_errors": errors,
    }


def assistant_trace_id(trace) -> str:
    info = (trace or {}).get("info") or {}
    return str(info.get("trace_id") or info.get("request_id") or "")

# COMMAND ----------

# DBTITLE 1,Traced request — one MLflow trace per request, the assistant's steps copied into it
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import mlflow
import requests
from databricks.sdk.core import Config
from mlflow.entities import SpanEvent, SpanStatus, SpanStatusCode

_cfg = Config()
HOST = _cfg.host.rstrip("/")
URL = f"{HOST}/serving-endpoints/{ENDPOINT}/invocations"
RETRYABLE = {429, 502, 503, 504}
BASELINE_WITH_DOCS = set()        # questions cited documents in the baseline (filled after the baseline)
TRACE_OPTION = [True]             # switched off if the endpoint rejects databricks_options.return_trace
_option_lock = threading.Lock()


class HttpStatusError(Exception):
    """HTTP error returned by the serving endpoint; recorded as an exception on the attempt step."""


def _error(code, message=""):
    return SpanStatus(SpanStatusCode.ERROR, message[:500]) if code == "ERROR" else SpanStatus(SpanStatusCode.OK)


def copy_assistant_steps(parent, spans: list, window_start: int, window_end: int):
    """Recreates the assistant's steps under `parent`, shifted into the HTTP call's time window (the assistant's clock
    is not the caller's). An empty retrieval step, logged OK by the assistant, is marked ERROR."""
    if not spans:
        return
    ids = {s["id"] for s in spans}
    starts = [s["start"] for s in spans if s["start"]]
    ends = [s["end"] for s in spans if s["end"]]
    t0, t1 = (min(starts), max(ends)) if starts and ends else (window_start, window_start)
    shift = window_start + max(0, (window_end - window_start) - (t1 - t0)) // 2 - t0
    children = {}
    for s in spans:
        children.setdefault(s["parent"] if s["parent"] in ids else None, []).append(s)
    created, queue = [], [(parent, s) for s in sorted(children.get(None, []), key=lambda x: x["start"])]
    while queue:
        live_parent, s = queue.pop(0)
        start = (s["start"] or t0) + shift
        span = mlflow.start_span_no_context(name=s["name"], span_type=s["type"], parent_span=live_parent,
                                            inputs=s["inputs"], attributes={**s["attributes"],
                                                                            "qualibot.source": "assistant trace"},
                                            start_time_ns=start)
        for e in s["events"]:
            span.add_event(SpanEvent(name=e["name"], timestamp=(e["time"] or s["start"] or t0) + shift,
                                     attributes=e["attributes"]))
        created.append((span, s, start))
        queue += [(span, c) for c in sorted(children.get(s["id"], []), key=lambda x: x["start"])]
    for span, s, start in reversed(created):
        if s["error"]:
            status = _error("ERROR", s["message"] or "error logged by the assistant")
        elif s["type"] == "RETRIEVER" and not documents_of(s["outputs"]):
            status = _error("ERROR", "0 documents retrieved (the assistant logged this step as OK)")
        else:
            status = _error("OK")
        span.end(outputs=s["outputs"], status=status, end_time_ns=max((s["end"] or t1) + shift, start))


def classify(rec: dict) -> str:
    code = rec["status_code"]
    if code == 429:
        return "http_429"
    if code is not None and code >= 500:
        return "http_5xx"
    if rec["error"] == "timeout":
        return "timeout"
    if code != 200:
        return "http_error" if code else "transport_error"
    if not str(rec["answer"] or "").strip():
        return "empty_answer"
    if rec["step_errors"]:
        return "step_error"
    if rec["retriever_steps"] and rec["retriever_docs"] == 0:
        return "retriever_empty"
    if not rec["refs"] and rec["phase"] != "baseline" and rec["question"] in BASELINE_WITH_DOCS:
        return "lost_documents"
    if not rec["refs"] and rec["sources_used"] is not True:
        return "no_documents"
    return "ok"


def _post(body: dict):
    return requests.post(URL, json=body, headers={**_cfg.authenticate(), "Content-Type": "application/json"},
                         timeout=TIMEOUT_S)


def attempt(root, rec: dict, n: int) -> float:
    """One HTTP call, traced as a step; returns the wait before a retry (0 = no retry)."""
    body = {"input": [{"role": "user", "content": rec["question"]}]}
    if TRACE_OPTION[0]:
        body["databricks_options"] = {"return_trace": True}
    span = mlflow.start_span_no_context(name=f"POST {ENDPOINT}", span_type="AGENT", parent_span=root, inputs=body,
                                        attributes={"attempt": n, "concurrency": rec["concurrency"]})
    t0 = time.time_ns()
    try:
        r = _post(body)
        if r.status_code == 400 and "databricks_options" in body and "option" in r.text.lower():
            with _option_lock:
                TRACE_OPTION[0] = False                      # endpoint without the trace option: resend without it
            body.pop("databricks_options")
            r = _post(body)
    except requests.Timeout as e:
        rec.update(status_code=None, error="timeout", latency_s=round((time.time_ns() - t0) / 1e9, 2))
        span.record_exception(e)
        span.end(outputs={"error": "timeout", "timeout_s": TIMEOUT_S})
        return _backoff(n, None)
    except Exception as e:
        rec.update(status_code=None, error=f"{type(e).__name__}: {str(e)[:300]}",
                   latency_s=round((time.time_ns() - t0) / 1e9, 2))
        span.record_exception(e)
        span.end(outputs={"error": rec["error"]})
        return 0.0
    t1 = time.time_ns()
    rec.update(status_code=r.status_code, retry_after=r.headers.get("Retry-After"), latency_s=round((t1 - t0) / 1e9, 2))
    span.set_attributes({"http.status_code": r.status_code, "latency_s": rec["latency_s"]})
    if not r.ok:
        rec["error"] = r.text[:500]
        span.record_exception(HttpStatusError(f"HTTP {r.status_code}: {r.text[:300]}"))
        span.end(outputs={"status_code": r.status_code, "retry_after": rec["retry_after"], "body": r.text[:2000]})
        return _backoff(n, rec["retry_after"]) if r.status_code in RETRYABLE else 0.0
    data = r.json()
    trace = (data.get("databricks_output") or {}).get("trace")
    spans = assistant_spans(trace)
    rec.update(error=None, **parse_answer(data), **retrieval_diagnostics(spans),
               assistant_trace_id=assistant_trace_id(trace))
    copy_assistant_steps(span, spans, t0, t1)
    span.end(outputs={"status_code": 200, "answer": rec["answer"], "cited_documents": rec["refs"],
                      "sources_used": rec["sources_used"]}, end_time_ns=t1)
    return 0.0


def _backoff(n: int, retry_after) -> float:
    if n > MAX_RETRIES:
        return 0.0
    try:
        return min(float(retry_after), BACKOFF_CAP_S)
    except (TypeError, ValueError):
        return min(2.0 ** (n - 1), BACKOFF_CAP_S) * (0.5 + random.random())


def send(question: str, phase: str, concurrency: int, idx: int) -> dict:
    """One request (with its retries) = one MLflow trace; returns its record."""
    rec = {"phase": phase, "concurrency": concurrency, "request_idx": idx, "question": question,
           "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "status_code": None,
           "retry_after": None, "error": None, "answer": None, "refs": [], "sources_used": None, "latency_s": None,
           "assistant_steps": 0, "retriever_steps": None, "retriever_docs": None, "reranker_step": None,
           "retriever_hidden": None, "step_errors": [], "assistant_trace_id": "", "attempts": 0,
           "throttled_attempts": 0, "backoff_s": 0.0}
    t_start = time.time()
    with mlflow.start_span(name="load_request", span_type="CHAIN") as root:
        root.set_inputs({"question": question, "phase": phase, "concurrency": concurrency, "request_idx": idx})
        while True:
            rec["attempts"] += 1
            wait = attempt(root, rec, rec["attempts"])
            rec["throttled_attempts"] += int(rec["status_code"] == 429)
            if not wait:
                break
            pause = mlflow.start_span_no_context(name="backoff", parent_span=root,
                                                 inputs={"after_attempt": rec["attempts"],
                                                         "retry_after": rec["retry_after"]})
            time.sleep(wait)
            pause.end(outputs={"waited_s": round(wait, 2)})
            rec["backoff_s"] += wait
        rec["total_s"] = round(time.time() - t_start, 2)
        rec["n_refs"] = len(rec["refs"])
        rec["outcome"] = outcome = classify(rec)
        detail = {"http_429": f"rate limited ({rec['throttled_attempts']} attempt(s) throttled)",
                  "step_error": "; ".join(rec["step_errors"])[:300],
                  "retriever_empty": "retrieval step returned 0 documents",
                  "lost_documents": "no document cited; the baseline answer cited documents",
                  "no_documents": "no document cited"}.get(outcome, str(rec["error"] or outcome)[:300])
        rec["detail"] = "" if outcome == "ok" else detail
        root.set_outputs({"outcome": outcome, "status_code": rec["status_code"], "answer": rec["answer"],
                          "cited_documents": rec["refs"], "retriever_docs": rec["retriever_docs"],
                          "attempts": rec["attempts"], "detail": rec["detail"]})
        if outcome != "ok":
            root.set_status(_error("ERROR", f"{outcome}: {detail}"))
        tags = {"test_id": TEST_ID, "endpoint": ENDPOINT, "phase": phase, "concurrency": str(concurrency),
                "request_idx": str(idx), "outcome": outcome, "status_code": str(rec["status_code"]),
                "attempts": str(rec["attempts"]), "latency_s": str(rec["latency_s"]),
                "retriever_docs": str(rec["retriever_docs"]), "citations": str(rec["n_refs"]),
                "assistant_trace_id": rec["assistant_trace_id"][:64]}
        mlflow.update_current_trace(tags=tags, request_preview=question[:500],
                                    response_preview=(rec["answer"] or f"{outcome}: {detail}")[:500])
        rec["trace_id"] = root.trace_id
    return rec


def run_level(phase: str, concurrency: int, questions: list) -> tuple:
    """Sends the questions with `concurrency` requests in flight; returns (records, wall time in s)."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(lambda a: send(a[1], phase, concurrency, a[0]), enumerate(questions)))
    wall = time.time() - t0
    for r in out:
        r["start_offset_s"] = round(datetime.fromisoformat(r["started_at"]).timestamp() - t0, 2)
    counts = {}
    for r in out:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print(f"{phase:>10} · {concurrency:>3} in flight · {len(out)} requests in {wall:.0f} s · {counts}")
    return out, wall

# COMMAND ----------

# DBTITLE 1,Load test — baseline, then one level after the other (one child run per level)
import os

import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID
# Traces are exported in the background: a worker thread takes its next request without waiting for the export,
# so the number of requests in flight is the one requested (flushed at the end of the test)
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
w.workspace.mkdirs(os.path.dirname(EXPERIMENT_PATH if EXPERIMENT_PATH.startswith("/Workspace")
                                   else "/Workspace" + EXPERIMENT_PATH))
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id
TEST_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
TEST_START_MS = int(time.time() * 1000)
LEVEL_RUNS, WALL, records = {}, {}, []


def level_metrics(frame: pd.DataFrame, wall: float) -> dict:
    ok200 = frame[frame["status_code"] == 200]
    silent = frame["outcome"].isin(["step_error", "retriever_empty", "lost_documents"]).sum()
    return {"requests": len(frame), "success_rate": float((frame["outcome"] == "ok").mean()),
            "http_429_rate": float((frame["outcome"] == "http_429").mean()),
            "error_rate": float(frame["outcome"].isin(["http_5xx", "timeout", "http_error", "transport_error"]).mean()),
            "silent_failure_rate": float(silent / len(ok200)) if len(ok200) else 0.0,
            "throughput_rps": len(frame) / wall if wall else 0.0,
            "latency_p50_s": float(ok200["latency_s"].median()) if len(ok200) else float("nan"),
            "latency_p95_s": float(ok200["latency_s"].quantile(.95)) if len(ok200) else float("nan"),
            "mean_citations": float(ok200["n_refs"].mean()) if len(ok200) else float("nan")}


plan = [("baseline", 1, list(QUESTIONS))] + [
    (f"load_{c}", c, [QUESTIONS[i % len(QUESTIONS)] for i in range(REQUESTS_PER_LEVEL)]) for c in LEVELS]
with mlflow.start_run(run_name=f"{ENDPOINT} · load test · {TEST_ID}") as parent_run:
    PARENT_RUN_ID = parent_run.info.run_id
    mlflow.set_tags({"endpoint": ENDPOINT, "test_id": TEST_ID, "kind": "load_test"})
    mlflow.log_params({"endpoint": ENDPOINT, "concurrency_levels": ",".join(map(str, LEVELS)),
                       "requests_per_level": REQUESTS_PER_LEVEL, "n_questions": len(QUESTIONS),
                       "questions_source": QUESTIONS_SOURCE, "max_retries": MAX_RETRIES, "timeout_s": TIMEOUT_S})
    for i, (phase, concurrency, questions) in enumerate(plan):
        if i > 1 and PAUSE_S:
            time.sleep(PAUSE_S)
        # Traces started in the worker threads are linked to the most recent active run: this level's run
        with mlflow.start_run(run_name=f"{phase} · {concurrency} in flight", nested=True) as level_run:
            mlflow.set_tags({"endpoint": ENDPOINT, "test_id": TEST_ID, "phase": phase})
            mlflow.log_params({"phase": phase, "concurrency": concurrency, "requests": len(questions)})
            out, wall = run_level(phase, concurrency, questions)
            metrics = level_metrics(pd.DataFrame(out), wall)
            mlflow.log_metrics(metrics)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v}, step=concurrency)
        LEVEL_RUNS[phase], WALL[phase] = level_run.info.run_id, wall
        records += out
        if phase == "baseline":
            BASELINE_WITH_DOCS.update(r["question"] for r in out if r["status_code"] == 200 and r["refs"])
mlflow.flush_trace_async_logging()
print(f"Assistant trace returned with the answer: {'yes' if TRACE_OPTION[0] else 'no (option rejected)'}")
print(f"MLflow: {EXPERIMENT_PATH} · run {PARENT_RUN_ID} · {len(records)} traces")

# COMMAND ----------

# DBTITLE 1,Outcome per request, logged on each trace as the assessment "outcome"
from mlflow.entities import AssessmentSource, AssessmentSourceType

df = pd.DataFrame(records)
df["test_id"], df["endpoint"] = TEST_ID, ENDPOINT
CODE_SOURCE = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="load_test")


def _log_outcome(r):
    for attempt_n in range(4):                        # a trace can take a few seconds to be stored
        try:
            mlflow.log_feedback(trace_id=r["trace_id"], name="outcome", value=r["outcome"], source=CODE_SOURCE,
                                rationale=r["detail"] or "HTTP 200, documents retrieved and cited")
            return True
        except Exception as e:
            error = str(e)[:200]
            time.sleep(2 ** attempt_n)
    return error


with ThreadPoolExecutor(max_workers=8) as pool:
    logged = list(pool.map(_log_outcome, df.to_dict("records")))
failed = [x for x in logged if x is not True]
print(f"✓ {len(logged) - len(failed)} assessments logged" + (f" · {len(failed)} failed: {failed[0]}" if failed else ""))

# COMMAND ----------

# DBTITLE 1,Report — outcomes, latency and throughput per level
from IPython.display import HTML as ipy_HTML, display as ipy_display

LEVEL_ORDER = [p for p, _, _ in plan]
GROUPS = {  # outcome → chart group (status colors: good, warning, serious, critical)
    "ok": "ok", "http_429": "rate limited (429)",
    "http_5xx": "other errors", "timeout": "other errors", "http_error": "other errors", "transport_error": "other errors",
    "step_error": "silent retrieval failure", "retriever_empty": "silent retrieval failure",
    "lost_documents": "silent retrieval failure", "empty_answer": "silent retrieval failure",
    "no_documents": "no documents (also in baseline)"}
GROUP_COLORS = {"ok": "#0ca30c", "rate limited (429)": "#fab219", "other errors": "#ec835a",
                "silent retrieval failure": "#d03b3b", "no documents (also in baseline)": "#898781"}
df["group"] = df["outcome"].map(GROUPS)


def _count(name):
    return ("outcome", lambda s: int((s == name).sum()))


summary = (df.groupby("phase").agg(
    concurrency=("concurrency", "first"), n=("outcome", "size"), ok=_count("ok"), http_429=_count("http_429"),
    http_5xx=_count("http_5xx"), timeouts=_count("timeout"),
    other_errors=("outcome", lambda s: int(s.isin(["http_error", "transport_error"]).sum())),
    empty_answers=_count("empty_answer"), step_error=_count("step_error"),
    retriever_empty=_count("retriever_empty"), lost_documents=_count("lost_documents"),
    no_documents=_count("no_documents"), attempts=("attempts", "sum"), mean_refs=("n_refs", "mean"))
    .reindex(LEVEL_ORDER).reset_index())
answered = df[df["status_code"] == 200].groupby("phase")["latency_s"]     # latency of answered requests only
summary["latency_p50_s"] = summary["phase"].map(answered.median())
summary["latency_p95_s"] = summary["phase"].map(answered.quantile(.95))
summary["wall_s"] = summary["phase"].map(WALL).round(1)
summary["throughput_rps"] = (summary["n"] / summary["wall_s"]).round(2)
summary["test_id"], summary["endpoint"] = TEST_ID, ENDPOINT
display(summary)

for r in summary.itertuples():
    issues = [f"{getattr(r, k)} {k}" for k in ["http_429", "http_5xx", "timeouts", "other_errors", "empty_answers",
                                                "step_error", "retriever_empty", "lost_documents"] if getattr(r, k)]
    print(f"{r.phase:>10} ({r.concurrency:>3} in flight): " + (", ".join(issues) if issues else "no anomaly")
          + f" · latency p50 {r.latency_p50_s:.1f} s, p95 {r.latency_p95_s:.1f} s · {r.throughput_rps} req/s")
throttled = summary[summary["http_429"] > 0]
silent = summary[(summary["step_error"] + summary["retriever_empty"] + summary["lost_documents"]) > 0]
clean = summary[(summary["ok"] + summary["no_documents"]) == summary["n"]]
print()
print(f"Rate limiting from {int(throttled['concurrency'].min())} requests in flight; Retry-After values: "
      f"{sorted(set(df.loc[df['status_code'] == 429, 'retry_after'].dropna())) or 'none sent'}"
      if len(throttled) else "No HTTP 429.")
print(f"Silent retrieval failures (HTTP 200, nothing retrieved) from {int(silent['concurrency'].min())} requests in flight."
      if len(silent) else "No silent retrieval failure: questions keep their documents under load.")
if len(clean):
    print(f"Highest level without any anomaly: {int(clean['concurrency'].max())} requests in flight.")

# COMMAND ----------

# DBTITLE 1,Charts — logged to the parent run (Artifacts: charts/)
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
groups_present = [g for g in GROUP_COLORS if g in set(df["group"])]
labels = [f"{p}\n({int(c)} in flight)" for p, c in zip(summary["phase"], summary["concurrency"])]

# 1. Outcomes per level: share of requests, stacked
fig1, ax = plt.subplots(figsize=(10, 0.6 * len(LEVEL_ORDER) + 1.6))
shares = (df.groupby(["phase", "group"]).size().unstack(fill_value=0).reindex(LEVEL_ORDER).fillna(0))
shares = shares.div(shares.sum(axis=1), axis=0)
left = pd.Series(0.0, index=shares.index)
for g in groups_present:
    ax.barh(labels, shares[g].values, left=left.values, color=GROUP_COLORS[g], label=g, height=0.6,
            edgecolor="#fcfcfb", linewidth=2)
    for y, (x0, v) in enumerate(zip(left.values, shares[g].values)):
        if v >= 0.08:
            ax.text(x0 + v / 2, y, f"{v:.0%}", ha="center", va="center", color=INK, fontsize=9)
    left += shares[g]
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Outcome of the requests per load level", loc="left", color=INK)
ax.legend(ncol=len(groups_present), loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False)
fig1.tight_layout()

# 2. Request timeline: one line per request, from start to end, colored by outcome
load = df[df["phase"] != "baseline"]
fig2, axes = plt.subplots(len(LEVELS), 1, figsize=(10, 2.2 * max(len(LEVELS), 1) + 0.8), squeeze=False, sharex=True)
for ax, phase in zip(axes[:, 0], [p for p in LEVEL_ORDER if p != "baseline"]):
    lv = load[load["phase"] == phase].sort_values("start_offset_s").reset_index(drop=True)
    for y, r in lv.iterrows():
        ax.hlines(y, r["start_offset_s"], r["start_offset_s"] + (r["total_s"] or 0), color=GROUP_COLORS[r["group"]],
                  linewidth=2)
    ax.set_ylabel(f"{int(lv['concurrency'].iloc[0]) if len(lv) else ''} in flight", color=MUTED)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
axes[-1, 0].set_xlabel("seconds since the start of the level")
axes[0, 0].set_title("Request timeline (one line per request, start → end)", loc="left", color=INK)
axes[-1, 0].legend(handles=[plt.Line2D([], [], color=GROUP_COLORS[g], linewidth=3, label=g) for g in groups_present],
                   ncol=len(groups_present), loc="upper left", bbox_to_anchor=(0, -0.35), frameon=False)
fig2.tight_layout()

# 3. Latency and throughput vs concurrency (two panels, one scale each)
fig3, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(10, 3.4))
x = summary["concurrency"]
for col, color, name in [("latency_p50_s", "#2a78d6", "p50"), ("latency_p95_s", "#eb6834", "p95")]:
    ax_l.plot(x, summary[col], color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax_l.annotate(name, (x.iloc[-1], summary[col].iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                  va="center")
ax_l.set_title("Latency (s)", loc="left", color=INK)
ax_l.set_xlabel("requests in flight")
ax_l.legend(frameon=False)
ax_t.plot(x, summary["throughput_rps"], color="#2a78d6", linewidth=2, marker="o", markersize=5)
ax_t.set_title("Throughput (requests/s)", loc="left", color=INK)
ax_t.set_xlabel("requests in flight")
for a in (ax_l, ax_t):
    a.set_xticks(list(x))
    a.grid(axis="y", color=GRID, linewidth=0.8)
    a.set_ylim(bottom=0)
fig3.tight_layout()

for name, fig in [("outcomes_by_level", fig1), ("request_timeline", fig2), ("latency_throughput", fig3)]:
    mlflow.MlflowClient().log_figure(PARENT_RUN_ID, fig, f"charts/{name}.png")
plt.show()

# COMMAND ----------

# DBTITLE 1,Failing requests — open them in the Traces tab by their trace id
import html as html_mod

TRACES_URL = f"{HOST}/ml/experiments/{EXPERIMENT_ID}/traces"
failing = df[~df["outcome"].isin(["ok", "no_documents"])].sort_values(["concurrency", "request_idx"])
if not len(failing):
    print("No failing request.")
else:
    rows = "".join(
        f"<tr><td>{html_mod.escape(r.phase)}</td><td>{r.request_idx}</td>"
        f"<td><span style='background:{GROUP_COLORS[r.group]};border-radius:4px;padding:1px 6px'>"
        f"{html_mod.escape(r.outcome)}</span></td><td>{r.status_code}</td><td>{r.attempts}</td><td>{r.latency_s}</td>"
        f"<td>{'' if r.retriever_docs is None or r.retriever_docs != r.retriever_docs else int(r.retriever_docs)}</td>"
        f"<td>{html_mod.escape(str(r.detail))[:200]}</td><td><code>{r.trace_id}</code></td></tr>"
        for r in failing.itertuples())
    ipy_display(ipy_HTML(
        f"<p><b>{len(failing)}</b> failing request(s) · <a href='{TRACES_URL}' target='_blank'>Traces tab</a> "
        f"(search a trace id, or filter <code>tags.outcome</code>)</p>"
        "<table style='font-family:sans-serif;font-size:12px;border-collapse:collapse'>"
        "<tr style='text-align:left'><th>phase</th><th>#</th><th>outcome</th><th>HTTP</th><th>attempts</th>"
        "<th>latency (s)</th><th>retrieved docs</th><th>detail</th><th>trace id</th></tr>" + rows + "</table>"))

# COMMAND ----------

# DBTITLE 1,The assistant's own experiment — which requests left a trace there (optional)
# The assistant logs its traces in its own experiment (Serving → endpoint → the agent's experiment id). A request
# rejected with HTTP 429 never reaches the assistant: it leaves no trace there, which is why no 429 appears in it.
if not ASSISTANT_EXPERIMENT_ID:
    print("assistant_experiment_id is empty: comparison skipped.")
else:
    server = mlflow.search_traces(locations=[ASSISTANT_EXPERIMENT_ID], max_results=5000, return_type="list",
                                  filter_string=f"trace.timestamp_ms >= {TEST_START_MS}")
    server_ids = {t.info.trace_id for t in server}
    df["in_assistant_experiment"] = df["assistant_trace_id"].map(lambda t: bool(t) and t in server_ids)
    view = (df.groupby(["phase", "outcome"]).agg(requests=("outcome", "size"),
                                                  traced_by_assistant=("in_assistant_experiment", "sum"))
            .reset_index())
    view["phase"] = pd.Categorical(view["phase"], LEVEL_ORDER, ordered=True)
    display(view.sort_values(["phase", "outcome"]))
    empty = 0
    for t in server:
        spans = [s for s in (t.data.spans if t.data else []) if str(s.span_type).upper() == "RETRIEVER"]
        empty += any(not documents_of(s.outputs) for s in spans)
    print(f"{len(server)} assistant traces since the start of the test ({len(df)} requests sent); "
          f"{empty} with an empty retrieval step, all logged with state "
          f"{sorted({str(t.info.state) for t in server}) or '—'}.")

# COMMAND ----------

# DBTITLE 1,Unity Catalog tables and request table on the run
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, LongType, StringType, StructField, StructType

S, B, I, D = StringType(), BooleanType(), LongType(), DoubleType()
REQUEST_COLUMNS = [("test_id", S), ("endpoint", S), ("phase", S), ("concurrency", I), ("request_idx", I),
                   ("started_at", S), ("question", S), ("outcome", S), ("detail", S), ("status_code", I),
                   ("retry_after", S), ("attempts", I), ("throttled_attempts", I), ("backoff_s", D),
                   ("latency_s", D), ("total_s", D), ("n_refs", I), ("refs", ArrayType(S)), ("sources_used", B),
                   ("retriever_steps", I), ("retriever_docs", I), ("reranker_step", B), ("retriever_hidden", B),
                   ("step_errors", ArrayType(S)), ("assistant_trace_id", S), ("trace_id", S), ("answer", S),
                   ("error", S)]
SUMMARY_COLUMNS = [("test_id", S), ("endpoint", S), ("phase", S), ("concurrency", I), ("n", I), ("ok", I),
                   ("http_429", I), ("http_5xx", I), ("timeouts", I), ("other_errors", I), ("empty_answers", I),
                   ("step_error", I), ("retriever_empty", I), ("lost_documents", I), ("no_documents", I),
                   ("attempts", I), ("mean_refs", D), ("latency_p50_s", D), ("latency_p95_s", D), ("wall_s", D),
                   ("throughput_rps", D)]


def to_rows(frame: pd.DataFrame, columns: list) -> list:
    def cell(v, t):
        if isinstance(t, ArrayType):
            return [str(x) for x in (v if isinstance(v, list) else [])]
        if v is None or (isinstance(v, float) and v != v):
            return None
        return {BooleanType: bool, LongType: int, DoubleType: float}.get(type(t), str)(v)
    return [tuple(cell(r.get(n), t) for n, t in columns) for r in frame.to_dict("records")]


df["answer"] = df["answer"].fillna("").str[:2000]
with mlflow.start_run(run_id=PARENT_RUN_ID):
    mlflow.log_table(df.drop(columns=["refs", "step_errors"]).astype(str), "requests.json")
    mlflow.log_table(summary.astype(str), "summary.json")
if WRITE_TABLES:
    for table, frame, columns in [(REQUESTS_TABLE, df, REQUEST_COLUMNS), (SUMMARY_TABLE, summary, SUMMARY_COLUMNS)]:
        (spark.createDataFrame(to_rows(frame, columns), StructType([StructField(n, t) for n, t in columns]))
              .write.mode("append").option("mergeSchema", "true").saveAsTable(table))
    print(f"✓ {len(df)} rows → {REQUESTS_TABLE} · {len(summary)} rows → {SUMMARY_TABLE} (test_id {TEST_ID})")
print(f"✓ request and summary tables logged to run {PARENT_RUN_ID}")
