# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Load Test
# MAGIC
# MAGIC Sends concurrent requests to a Knowledge Assistant serving endpoint and looks for the two failure modes seen under
# MAGIC load:
# MAGIC - **rate limiting**: HTTP 429 (and other HTTP errors, timeouts), with the `Retry-After` header when present;
# MAGIC - **silent retrieval failures**: HTTP 200 with a well-formed answer, but no document retrieved — no citation, a
# MAGIC   "not found" answer, `sources_used = false`, or an empty retrieval step when the endpoint returns its trace.
# MAGIC
# MAGIC ### Method
# MAGIC 1. **Baseline**: every question is sent once, one request at a time. It records which questions normally come back
# MAGIC    with documents.
# MAGIC 2. **Load levels** (e.g. 10 then 30 concurrent requests): the same questions are sent again, cycled, with that many
# MAGIC    requests in flight. A question that had documents in the baseline and comes back without any is a
# MAGIC    `lost_documents` request: the retriever failed silently.
# MAGIC 3. No retry: every 429 is counted, never hidden.
# MAGIC
# MAGIC ### Outputs
# MAGIC | Where | Content |
# MAGIC |---|---|
# MAGIC | Notebook | summary per level, anomalies, latency, every failing request |
# MAGIC | `ka_load_test_requests` | one row per request (status, latency, documents, anomaly) |
# MAGIC | `ka_load_test_summary` | one row per test and level |
# MAGIC | MLflow experiment | one run per test: parameters, metrics per level, the request table |
# MAGIC
# MAGIC ⚠️ The test loads a real endpoint: run it outside working hours on an endpoint used by people, and keep in mind that
# MAGIC every request is billed like a normal question.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("endpoint", "ka-7679a56e-endpoint")                 # serving endpoint to test (qualibot_ALL_v2)
dbutils.widgets.text("concurrency_levels", "10,30")                      # concurrent requests per load level
dbutils.widgets.text("requests_per_level", "60")                         # requests sent at each load level
dbutils.widgets.dropdown("questions_source", "fixed", ["fixed", "logs"]) # fixed list below, or a sample of real questions
dbutils.widgets.text("n_questions", "10")                                # questions used (logs source)
dbutils.widgets.text("timeout_s", "180")                                 # per-request timeout
dbutils.widgets.text("pause_between_levels_s", "60")                     # lets rate limits reset between levels
dbutils.widgets.dropdown("request_trace", "true", ["true", "false"])     # ask the endpoint for its trace (retrieval step)
dbutils.widgets.dropdown("write_results", "true", ["true", "false"])     # Unity Catalog tables + MLflow run
dbutils.widgets.text("output_schema", "uat_proj.qualibot")
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")
dbutils.widgets.text("experiment_path", "/Shared/qualibot-load-tests")

ENDPOINT = dbutils.widgets.get("endpoint").strip()
LEVELS = [int(x) for x in dbutils.widgets.get("concurrency_levels").split(",") if x.strip()]
REQUESTS_PER_LEVEL = int(dbutils.widgets.get("requests_per_level"))
QUESTIONS_SOURCE = dbutils.widgets.get("questions_source")
N_QUESTIONS = int(dbutils.widgets.get("n_questions"))
TIMEOUT_S = float(dbutils.widgets.get("timeout_s"))
PAUSE_S = float(dbutils.widgets.get("pause_between_levels_s"))
REQUEST_TRACE = dbutils.widgets.get("request_trace") == "true"
WRITE_RESULTS = dbutils.widgets.get("write_results") == "true"
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
REQUESTS_TABLE = f"{OUTPUT_SCHEMA}.ka_load_test_requests"
SUMMARY_TABLE = f"{OUTPUT_SCHEMA}.ka_load_test_summary"

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
NOT_FOUND_PHRASES = ["pas trouvé", "aucune information", "ne contient pas", "ne dispose pas", "pas d'information",
                     "n'ai pas trouvé", "not found", "no information", "does not contain", "could not find",
                     "unable to find", "no relevant document"]
print(f"{ENDPOINT} · baseline then levels {LEVELS} · {REQUESTS_PER_LEVEL} requests per level · "
      f"questions: {QUESTIONS_SOURCE} · trace requested: {REQUEST_TRACE}")

# COMMAND ----------

# DBTITLE 1,Questions
import re

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
else:
    QUESTIONS = FIXED_QUESTIONS[:N_QUESTIONS]
print(f"{len(QUESTIONS)} questions")
for q in QUESTIONS:
    print(" -", q[:120])

# COMMAND ----------

# DBTITLE 1,Request — one call to the endpoint, without retry; answer, documents, retrieval step
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from databricks.sdk.core import Config

_cfg = Config()
URL = f"{_cfg.host.rstrip('/')}/serving-endpoints/{ENDPOINT}/invocations"
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)
TRACE_SUPPORTED = [REQUEST_TRACE]      # switched off if the endpoint rejects the trace option


def _walk(x):
    """Every dict nested in a JSON value."""
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def parse_response(data: dict) -> dict:
    """Answer text, cited documents, sources_used flag and retrieval step of a Responses-format answer."""
    texts, refs, sources_used = [], [], None
    for item in (data.get("output") or []):
        for c in (item.get("content") or []) if isinstance(item, dict) else []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                texts.append(c.get("text") or "")
                for a in c.get("annotations") or []:
                    m = _REF_IN_URL.search(str(a.get("url") or a.get("title") or ""))
                    if m:
                        refs.append(m.group(1))
    for d in _walk(data):
        if isinstance(d.get("custom_outputs"), dict) and "sources_used" in d["custom_outputs"]:
            sources_used = bool(d["custom_outputs"]["sources_used"])
    if not texts and data.get("choices"):
        texts = [data["choices"][0]["message"].get("content") or ""]
    answer = "\n".join(texts)
    if not refs:
        refs = _REF_IN_URL.findall(answer)

    # Retrieval step of the returned trace (spans of type RETRIEVER), when the endpoint returns it
    retriever_spans = retriever_docs = None
    trace = (data.get("databricks_output") or {}).get("trace")
    if trace:
        spans = (trace.get("data") or {}).get("spans") or []
        retrievers = [s for s in spans if "RETRIEVER" in json.dumps(s.get("attributes") or {})
                      or str(s.get("span_type") or "").upper() == "RETRIEVER"]
        retriever_spans = len(retrievers)
        retriever_docs = 0
        for s in retrievers:
            out = (s.get("attributes") or {}).get("mlflow.spanOutputs", s.get("outputs"))
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except json.JSONDecodeError:
                    out = None
            retriever_docs += len(out) if isinstance(out, list) else 0
    return {"answer": answer, "refs": sorted(set(refs)), "sources_used": sources_used,
            "retriever_spans": retriever_spans, "retriever_docs": retriever_docs}


def call(question: str, phase: str, concurrency: int, idx: int) -> dict:
    body = {"input": [{"role": "user", "content": question}]}
    if TRACE_SUPPORTED[0]:
        body["databricks_options"] = {"return_trace": True}
    rec = {"phase": phase, "concurrency": concurrency, "request_idx": idx, "question": question,
           "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "status_code": None,
           "retry_after": None, "error": None, "answer": None, "refs": [], "n_refs": 0, "sources_used": None,
           "retriever_spans": None, "retriever_docs": None}
    t0 = time.time()
    try:
        r = requests.post(URL, json=body, headers={**_cfg.authenticate(), "Content-Type": "application/json"},
                          timeout=TIMEOUT_S)
        if r.status_code == 400 and "databricks_options" in body and "option" in r.text.lower():
            TRACE_SUPPORTED[0] = False             # endpoint without trace option: resend without it
            return call(question, phase, concurrency, idx)
        rec["status_code"], rec["retry_after"] = r.status_code, r.headers.get("Retry-After")
        if r.ok:
            rec.update(parse_response(r.json()))
            rec["n_refs"] = len(rec["refs"])
        else:
            rec["error"] = r.text[:500]
    except requests.Timeout:
        rec["error"] = "timeout"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    rec["latency_s"] = round(time.time() - t0, 2)
    return rec


def run_level(phase: str, concurrency: int, questions: list) -> list:
    """Sends the questions with `concurrency` requests in flight; returns one record per request."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(lambda a: call(a[1], phase, concurrency, a[0]), enumerate(questions)))
    print(f"{phase}: {len(out)} requests at concurrency {concurrency} in {time.time() - t0:.0f} s")
    return out

# COMMAND ----------

# DBTITLE 1,Run — baseline (one request at a time), then each load level
TEST_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
records = run_level("baseline", 1, QUESTIONS)
for level in LEVELS:
    time.sleep(PAUSE_S)
    cycled = [QUESTIONS[i % len(QUESTIONS)] for i in range(REQUESTS_PER_LEVEL)]
    records += run_level(f"load_{level}", level, cycled)
print(f"Trace returned by the endpoint: {'yes' if TRACE_SUPPORTED[0] else 'no (option rejected)'}")

# COMMAND ----------

# DBTITLE 1,Anomalies — one label per request, compared with the baseline
import pandas as pd

df = pd.DataFrame(records)
df["test_id"], df["endpoint"] = TEST_ID, ENDPOINT
df["says_not_found"] = df["answer"].fillna("").str.lower().map(lambda a: any(p in a for p in NOT_FOUND_PHRASES))
base = df[(df["phase"] == "baseline") & (df["status_code"] == 200)]
HAD_DOCS = set(base.loc[base["n_refs"] > 0, "question"])


def anomaly(r) -> str:
    if r.status_code == 429:
        return "http_429"
    if r.status_code is not None and r.status_code >= 500:
        return "http_5xx"
    if r.error == "timeout":
        return "timeout"
    if r.status_code != 200:
        return "http_other" if r.status_code else "transport_error"
    if not str(r.answer or "").strip():
        return "empty_answer"
    if r.retriever_spans is not None and r.retriever_spans > 0 and r.retriever_docs == 0:
        return "retriever_empty"
    if r.n_refs == 0 and r.question in HAD_DOCS and r.phase != "baseline":
        return "lost_documents"
    if r.n_refs == 0 and r.sources_used is not True:
        return "no_documents"
    return "ok"


df["anomaly"] = df.apply(anomaly, axis=1)
LEVEL_ORDER = ["baseline"] + [f"load_{c}" for c in LEVELS]
summary = (df.groupby("phase").agg(
    concurrency=("concurrency", "first"), n=("anomaly", "size"),
    http_429=("anomaly", lambda s: int((s == "http_429").sum())),
    http_5xx=("anomaly", lambda s: int((s == "http_5xx").sum())),
    timeouts=("anomaly", lambda s: int((s == "timeout").sum())),
    other_errors=("anomaly", lambda s: int(s.isin(["http_other", "transport_error"]).sum())),
    empty_answers=("anomaly", lambda s: int((s == "empty_answer").sum())),
    retriever_empty=("anomaly", lambda s: int((s == "retriever_empty").sum())),
    lost_documents=("anomaly", lambda s: int((s == "lost_documents").sum())),
    no_documents=("anomaly", lambda s: int((s == "no_documents").sum())),
    not_found_answers=("says_not_found", "sum"),
    mean_refs=("n_refs", "mean"),
    latency_p50_s=("latency_s", "median"),
    latency_p95_s=("latency_s", lambda s: float(s.quantile(.95))))
    .reindex(LEVEL_ORDER).reset_index())
summary["test_id"], summary["endpoint"] = TEST_ID, ENDPOINT
display(summary)

# COMMAND ----------

# DBTITLE 1,Diagnosis — what failed, from which concurrency level
for r in summary.itertuples():
    issues = [f"{getattr(r, k)} {k}" for k in ["http_429", "http_5xx", "timeouts", "other_errors", "empty_answers",
                                                "retriever_empty", "lost_documents", "no_documents"] if getattr(r, k)]
    print(f"{r.phase:>10} (x{r.concurrency}): " + (", ".join(issues) if issues else "no anomaly")
          + f" · latency p50 {r.latency_p50_s:.1f} s, p95 {r.latency_p95_s:.1f} s")
rate_limited = summary[summary["http_429"] > 0]
silent = summary[(summary["lost_documents"] + summary["retriever_empty"]) > 0]
print()
print(f"Rate limiting starts at concurrency {int(rate_limited['concurrency'].min())}; Retry-After seen: "
      f"{sorted(set(df.loc[df['status_code'] == 429, 'retry_after'].dropna()))}"
      if len(rate_limited) else "No HTTP 429.")
print(f"Silent retrieval failures (HTTP 200 without documents) start at concurrency {int(silent['concurrency'].min())}."
      if len(silent) else "No silent retrieval failure: questions keep their documents under load.")
failing = df[df["anomaly"] != "ok"]
if len(failing):
    display(failing[["phase", "request_idx", "anomaly", "status_code", "retry_after", "latency_s", "n_refs",
                     "sources_used", "retriever_docs", "question", "error"]].sort_values(["phase", "request_idx"]))

# COMMAND ----------

# DBTITLE 1,Results — Unity Catalog tables and MLflow run
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, LongType, StringType, StructField,
                               StructType)

S, B, I, D = StringType(), BooleanType(), LongType(), DoubleType()
REQUEST_COLUMNS = [("test_id", S), ("endpoint", S), ("phase", S), ("concurrency", I), ("request_idx", I),
                   ("started_at", S), ("question", S), ("status_code", I), ("retry_after", S), ("latency_s", D),
                   ("anomaly", S), ("n_refs", I), ("refs", ArrayType(S)), ("sources_used", B),
                   ("retriever_spans", I), ("retriever_docs", I), ("says_not_found", B), ("answer", S), ("error", S)]
SUMMARY_COLUMNS = [("test_id", S), ("endpoint", S), ("phase", S), ("concurrency", I), ("n", I), ("http_429", I),
                   ("http_5xx", I), ("timeouts", I), ("other_errors", I), ("empty_answers", I),
                   ("retriever_empty", I), ("lost_documents", I), ("no_documents", I), ("not_found_answers", I),
                   ("mean_refs", D), ("latency_p50_s", D), ("latency_p95_s", D)]


def to_rows(frame: pd.DataFrame, columns: list) -> list:
    def cell(v, t):
        if isinstance(t, ArrayType):
            return [str(x) for x in (v if isinstance(v, list) else [])]
        if v is None or (isinstance(v, float) and v != v):
            return None
        return {BooleanType: bool, LongType: int, DoubleType: float}.get(type(t), str)(v)
    return [tuple(cell(r[n], t) for n, t in columns) for r in frame.to_dict("records")]


if WRITE_RESULTS:
    df["answer"] = df["answer"].fillna("").str[:2000]
    for table, frame, columns in [(REQUESTS_TABLE, df, REQUEST_COLUMNS), (SUMMARY_TABLE, summary, SUMMARY_COLUMNS)]:
        (spark.createDataFrame(to_rows(frame, columns), StructType([StructField(n, t) for n, t in columns]))
              .write.mode("append").option("mergeSchema", "true").saveAsTable(table))
    print(f"✓ {len(df)} rows → {REQUESTS_TABLE} · {len(summary)} rows → {SUMMARY_TABLE} (test_id {TEST_ID})")

    import mlflow
    from databricks.sdk import WorkspaceClient

    WorkspaceClient().workspace.mkdirs(EXPERIMENT_PATH.rsplit("/", 1)[0].replace("/Shared", "/Workspace/Shared", 1))
    mlflow.set_experiment(EXPERIMENT_PATH)
    with mlflow.start_run(run_name=f"{ENDPOINT} · load test · {TEST_ID}"):
        mlflow.log_params({"endpoint": ENDPOINT, "concurrency_levels": ",".join(map(str, LEVELS)),
                           "requests_per_level": REQUESTS_PER_LEVEL, "n_questions": len(QUESTIONS),
                           "questions_source": QUESTIONS_SOURCE, "trace_returned": TRACE_SUPPORTED[0]})
        for r in summary.to_dict("records"):
            mlflow.log_metrics({f"{r['phase']}/{k}": float(r[k]) for k, _ in SUMMARY_COLUMNS[4:] if r[k] == r[k]})
        mlflow.log_table(df.drop(columns=["refs"]).astype(str), "requests.json")
    print(f"✓ MLflow run logged in {EXPERIMENT_PATH}")
