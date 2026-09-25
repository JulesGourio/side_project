# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Evaluation
# MAGIC
# MAGIC Evaluates a Knowledge Assistant endpoint against the golden evaluation dataset stored in Unity Catalog,
# MAGIC with MLflow GenAI evaluation. Everything is visible in the MLflow experiment:
# MAGIC
# MAGIC | Where (MLflow UI) | What |
# MAGIC |---|---|
# MAGIC | **Runs** | one run per evaluated endpoint and repetition, with parameters, aggregated metrics and report tables |
# MAGIC | **Traces** | one trace per case: the assistant call, the answer, and a `RETRIEVER` step holding the excerpts of the documents the answer cites; every judge verdict is attached to the trace with its rationale |
# MAGIC | **Judges** | the LLM judges used by this notebook, registered in the experiment (not scheduled: no background cost) |
# MAGIC | **Datasets** | the golden dataset the runs were evaluated on |
# MAGIC
# MAGIC ### Quality dimensions
# MAGIC | Axis | Judge | Type | Question answered |
# MAGIC |---|---|---|---|
# MAGIC | Correctness | `correctness` | built-in | Does the answer contain the expected facts (or match the expected answer)? |
# MAGIC | | `fact_coverage` | custom LLM | full / partial / none — finer than a yes/no |
# MAGIC | | `fact_contradiction` | custom LLM | Does the answer contradict an expected fact? |
# MAGIC | | `refusal_handling` | custom LLM | When the expected behaviour is a refusal ("not in the documentation", out of scope), does the assistant refuse without inventing? |
# MAGIC | Faithfulness | `retrieval_groundedness` | built-in | Is the answer supported by the excerpts of the documents it cites? |
# MAGIC | Retrieval | `retrieval_sufficiency` | built-in | Do those excerpts contain what the expected answer needs? |
# MAGIC | | `document_recall` | code | Share of the expected documents the assistant returned or cited |
# MAGIC | | `reference_integrity` | code | Every cited document code exists (typos such as missing zero padding are resolved, not penalised) |
# MAGIC | Compliance | `expectations_guidelines` | built-in | Case-specific guidelines (language, mandatory citation, …) |
# MAGIC | | `relevance_to_query` | built-in | Does the answer address the question? |
# MAGIC | Operations | `call_ok`, `latency_s` | code | Endpoint availability and response time |
# MAGIC
# MAGIC ### How to use
# MAGIC 1. Run the setup cells, then **Case selection** with `run_eval=false` to review the selected cases and the estimated number of calls.
# MAGIC 2. Set `run_eval=true` (start with `sample_n=5`), then read the **Report**.
# MAGIC 3. Use **Human labels** to rate a few answers: the agreement between you and the judges tells whether the scores can be trusted,
# MAGIC    and the same labels can align the custom judges (**Judge alignment**).

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

NEEDED = {"mlflow": (3, 4)}   # make_judge, judge registration and alignment


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
dbutils.widgets.text("dataset_name", "uat_landingzone.qualibot.qualibot_eval_golden")
dbutils.widgets.text("experiment_path",
                     "/Workspace/Users/jules.gourio.external@latecoere.aero/qualibot-traces/trace_eval_all_v2")
dbutils.widgets.text("endpoints", "ka-7679a56e-endpoint")          # comma-separated serving endpoints to evaluate
dbutils.widgets.text("sql_warehouse_id", "5890912c31867b77")        # required to read traces stored in Unity Catalog
dbutils.widgets.text("judge_endpoint", "databricks-gpt-5-6-luna")   # empty = Databricks-managed judge model
dbutils.widgets.text("sample_n", "5")                               # empty = every case of the dataset
dbutils.widgets.text("case_ids", "")                                # comma-separated case ids (overrides sample_n)
dbutils.widgets.text("repeats", "1")                                # >1 measures the assistant's variability
dbutils.widgets.dropdown("run_eval", "false", ["true", "false"])

DATASET_NAME = dbutils.widgets.get("dataset_name").strip()
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
ENDPOINTS = [e.strip() for e in dbutils.widgets.get("endpoints").split(",") if e.strip()]
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id").strip()
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint").strip()
SAMPLE_N = int(dbutils.widgets.get("sample_n")) if dbutils.widgets.get("sample_n").strip() else None
CASE_IDS = [c.strip() for c in dbutils.widgets.get("case_ids").split(",") if c.strip()]
REPEATS = max(1, int(dbutils.widgets.get("repeats") or 1))
RUN_EVAL = dbutils.widgets.get("run_eval") == "true"

# Retrieval context used to verify the answers (documents cited by the assistant)
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None          # source table of the index; None = read from the index definition
EXCERPTS_PER_CASE = 12
EXCERPT_MAX_CHARS = 1500

# Optional case metadata (question type, difficulty, expected answerability) written by the dataset builder
GOLDEN_CACHE_TABLE = "uat_landingzone.qualibot.qualibot_eval_cache"

MAX_PARALLEL_CALLS = 3           # capacity limit of the Knowledge Assistant endpoints
SEED = 42
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX"]

# COMMAND ----------

# DBTITLE 1,Connections — experiment, dataset, cases
import json
import math
import os
import re
import time

import mlflow
import mlflow.genai.datasets as gdatasets
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import display

w = WorkspaceClient()
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID          # traces are stored in Unity Catalog
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(MAX_PARALLEL_CALLS)
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id
eval_ds = gdatasets.get_dataset(name=DATASET_NAME)

CASES = eval_ds.to_df().reset_index(drop=True)
CASES["question"] = CASES["inputs"].map(lambda i: i["messages"][-1]["content"])
CASES["case_id"] = (CASES["dataset_record_id"].astype(str) if "dataset_record_id" in CASES.columns
                    else CASES.index.astype(str))


def _golden_metadata() -> pd.DataFrame:
    """Case metadata from the dataset builder cache (optional)."""
    try:
        rows = (spark.table(GOLDEN_CACHE_TABLE).filter("stage = 'golden_final'")
                .select("question_id", "payload").collect())
    except Exception:
        return pd.DataFrame(columns=["question"])
    recs = [{**json.loads(r.payload), "question_id": int(r.question_id)} for r in rows]
    keep = ["question_id", "question", "intent", "difficulty", "final_answerability", "ka_verdict"]
    return pd.DataFrame(recs)[[c for c in keep if c in pd.DataFrame(recs).columns]]


meta = _golden_metadata()
if len(meta):
    CASES = CASES.merge(meta, on="question", how="left")
    CASES["case_id"] = CASES["question_id"].map(lambda q: str(int(q)) if pd.notna(q) else None).fillna(CASES["case_id"])
for col in ["intent", "difficulty", "final_answerability", "ka_verdict"]:
    if col not in CASES.columns:
        CASES[col] = None
CASE_BY_QUESTION = dict(zip(CASES["question"], CASES["case_id"]))
print(f"Experiment: {EXPERIMENT_PATH} (id {EXPERIMENT_ID})")
print(f"Dataset   : {DATASET_NAME} · {len(CASES)} cases · metadata {'available' if len(meta) else 'not available'}")

# COMMAND ----------

# DBTITLE 1,Document references — keys insensitive to language suffix, separators and zero padding
_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|txt)$", re.I)
_LANG = re.compile(r"[-_ ](%s)$" % "|".join(LANG_SUFFIXES), re.I)
_CODE = re.compile(r"^(?=.*\d)[A-Z][A-Z0-9_\-]{2,28}( (%s))?$" % "|".join(LANG_SUFFIXES))
_BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def _strip(s) -> str:
    return _LANG.sub("", _EXT.sub("", str(s).strip().split("/")[-1]))


def base_ref(s) -> str:
    """Document key: PRLAT538_FR and prlat-538 GB → PRLAT538; IN_APO_006 (typo present in some documents)
    and IN_APO_0006 → INAPO6. Language variants of a document therefore share one key."""
    groups = re.findall(r"[A-Za-z]+|\d+", _strip(s).upper())
    return "".join(str(int(g)) if g.isdigit() else g for g in groups)


def exact_ref(s) -> str:
    """Exact form of a code (separators, case and language suffix ignored, zero padding kept)."""
    return re.sub(r"[^A-Z0-9]", "", _strip(s).upper())


def code_like(text) -> set:
    """Document codes cited in an answer: **REF** in bold and ?ref= parameters of links."""
    text = str(text or "")
    return {c.strip() for c in _BOLD.findall(text) + _REF_IN_URL.findall(text) if _CODE.match(c.strip().upper())}


REFS_BY_BASE, EXACT_REFS = {}, set()
try:
    _src = REF_SOURCE_TABLE or w.vector_search_indexes.get_index(VS_INDEX).delta_sync_index_spec.source_table
    for r in spark.table(_src).select("REF").distinct().collect():
        if r.REF:
            REFS_BY_BASE.setdefault(base_ref(r.REF), set()).add(r.REF)
            EXACT_REFS.add(exact_ref(r.REF))
    print(f"Document index: {len(REFS_BY_BASE)} documents ({_src})")
except Exception as e:
    print(f"⚠️ Document index unavailable — retrieval excerpts and reference checks disabled: {str(e)[:200]}")


def resolve_refs(refs) -> list:
    """Cited codes → exact REF values of the index (all language variants)."""
    return sorted({real for r in refs for real in REFS_BY_BASE.get(base_ref(r), set())})


def refs_from_response(raw) -> set:
    """Documents returned by the assistant in its raw response (citations carrying a title or a ?ref= URL)."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            url, title = x.get("url"), x.get("title")
            if isinstance(url, str) and _REF_IN_URL.search(url):
                out.add(_REF_IN_URL.search(url).group(1))
            elif isinstance(title, str) and title.strip():
                out.add(title.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(raw)
    return {r for r in out if base_ref(r) in REFS_BY_BASE} if REFS_BY_BASE else out


def classify_refs(cited, excerpt_text: str) -> dict:
    """approximate: resolved typos (zero padding) · unindexed: absent from the index but mentioned in the excerpts
    (document outside the corpus) · unverified: found nowhere (possibly invented)."""
    out = {"approximate": [], "unindexed": [], "unverified": []}
    if not REFS_BY_BASE:
        return out
    excerpt_keys = {base_ref(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,28}", excerpt_text or "")}
    for c in cited:
        key = base_ref(c)
        if key in REFS_BY_BASE:
            if exact_ref(c) not in EXACT_REFS:
                out["approximate"].append(f"{c} → {sorted(REFS_BY_BASE[key])[0]}")
        elif key in excerpt_keys:
            out["unindexed"].append(c)
        else:
            out["unverified"].append(c)
    return out

# COMMAND ----------

# DBTITLE 1,Traced application — assistant call and retrieval of the cited documents' excerpts
from mlflow.entities import Document


def _extract_text(resp) -> str:
    if isinstance(resp, dict):
        if resp.get("output"):
            texts = [c.get("text", "") for item in resp["output"] if isinstance(item, dict)
                     for c in (item.get("content") or []) if isinstance(c, dict) and c.get("type") in ("output_text", "text")]
            if texts:
                return "\n".join(texts)
        if resp.get("choices"):
            return resp["choices"][0]["message"]["content"]
        if resp.get("messages"):
            return resp["messages"][-1].get("content")
    return json.dumps(resp, ensure_ascii=False)[:4000]


@mlflow.trace(name="knowledge_assistant", span_type="AGENT")
def call_assistant(endpoint: str, messages: list) -> dict:
    """Raw call to the Knowledge Assistant serving endpoint (Responses format, then Chat format)."""
    last_error = None
    for body in ({"input": messages}, {"messages": messages}):
        try:
            return w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        except Exception as e:
            last_error = e
    raise last_error


@mlflow.trace(name="cited_document_excerpts", span_type="RETRIEVER")
def cited_document_excerpts(query: str, refs: list) -> list:
    """Excerpts of the documents the answer relies on, most related to the question and the answer.
    Exposed as a RETRIEVER step so that retrieval judges can assess them and they are readable in the trace."""
    real = resolve_refs(refs)
    if not real:
        return []
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX, columns=VS_COLUMNS, query_text=query[:2000], query_type="HYBRID",
        num_results=EXCERPTS_PER_CASE, filters_json=json.dumps({"REF": real}))
    cols = [c.name for c in res.manifest.columns]
    rows = [dict(zip(cols, r)) for r in ((res.result.data_array if res.result else None) or [])]
    return [Document(id=f"{r.get('REF')}#{i}", page_content=str(r.get("chunk_text") or "")[:EXCERPT_MAX_CHARS],
                     metadata={"doc_uri": r.get("REF"), "section": str(r.get("semantic_headers") or "")[:200]})
            for i, r in enumerate(rows)]


def make_predict_fn(endpoint: str):
    @mlflow.trace(name="qualibot_turn", span_type="AGENT")
    def predict_fn(messages):
        question = messages[-1]["content"]
        tags = {"case_id": str(CASE_BY_QUESTION.get(question, "")), "endpoint": endpoint}
        t0 = time.time()
        try:
            raw = call_assistant(endpoint, messages)
        except Exception as e:
            mlflow.update_current_trace(tags={**tags, "call_ok": "false", "error": str(e)[:500]})
            cited_document_excerpts(question, [])
            return ""
        answer = _extract_text(raw) or ""
        returned = sorted(refs_from_response(raw))
        cited = sorted(code_like(answer))
        docs = cited_document_excerpts(f"{question}\n{answer[:800]}", returned + cited)
        refs = classify_refs(cited, "\n".join(d.page_content for d in docs))
        mlflow.update_current_trace(tags={
            **tags, "call_ok": "true", "latency_s": f"{time.time() - t0:.2f}",
            "returned_refs": json.dumps(returned, ensure_ascii=False), "cited_refs": json.dumps(cited, ensure_ascii=False),
            "approximate_refs": json.dumps(refs["approximate"], ensure_ascii=False),
            "unindexed_refs": json.dumps(refs["unindexed"], ensure_ascii=False),
            "unverified_refs": json.dumps(refs["unverified"], ensure_ascii=False)})
        return answer
    return predict_fn

# COMMAND ----------

# DBTITLE 1,Judges — built-in, custom LLM (make_judge) and code-based scorers
from typing import Literal

from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (Correctness, ExpectationsGuidelines, RelevanceToQuery, RetrievalGroundedness,
                                  RetrievalSufficiency, scorer)

FACT_COVERAGE_INSTRUCTIONS = """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Compare the answer in {{ outputs }} to the question in {{ inputs }} and to the expected facts in
{{ expectations }}. Be strict on substance and tolerant on wording (paraphrases, synonyms, other language).
Return:
- full: every expected fact is present;
- partial: some expected facts are present, others are missing or imprecise;
- none: no expected fact is present;
- not_applicable: the expectations contain no expected_facts (the expected behaviour is a refusal)."""

FACT_CONTRADICTION_INSTRUCTIONS = """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Check whether the answer in {{ outputs }} states anything incompatible with the expected facts in
{{ expectations }} for the question in {{ inputs }}: a different value, threshold, deadline, responsible role or document.
Missing facts are NOT contradictions. Return no_contradiction, contradiction, or not_applicable when the expectations
contain no expected_facts."""

REFUSAL_HANDLING_INSTRUCTIONS = """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Apply this judge only when the expectations in {{ expectations }} contain no expected_facts: the expected
behaviour is then to state that the information is not in the documentation, or to decline an out-of-scope request.
For the question in {{ inputs }} and the answer in {{ outputs }}, return:
- correct_refusal: the answer clearly says the information is not available (or declines) and does not fabricate an
  answer; mentioning what was found nearby or asking for clarification is acceptable;
- answered_anyway: the answer fabricates an answer to the question;
- not_applicable: the expectations contain expected_facts."""


def build_llm_judges(model):
    kw = {"model": model} if model else {}
    return [
        Correctness(**kw), RelevanceToQuery(**kw), ExpectationsGuidelines(**kw),
        RetrievalGroundedness(**kw), RetrievalSufficiency(**kw),
        make_judge(name="fact_coverage", instructions=FACT_COVERAGE_INSTRUCTIONS,
                   feedback_value_type=Literal["full", "partial", "none", "not_applicable"], **kw),
        make_judge(name="fact_contradiction", instructions=FACT_CONTRADICTION_INSTRUCTIONS,
                   feedback_value_type=Literal["no_contradiction", "contradiction", "not_applicable"], **kw),
        make_judge(name="refusal_handling", instructions=REFUSAL_HANDLING_INSTRUCTIONS,
                   feedback_value_type=Literal["correct_refusal", "answered_anyway", "not_applicable"], **kw),
    ]


def _tags(trace) -> dict:
    return (trace.info.tags if trace else None) or {}


def _clean_uri(u) -> str:
    return re.sub(r"^\s*REF\s*:\s*", "", str(u))


@scorer
def document_recall(expectations, trace):
    """Share of the expected documents returned or cited by the assistant (language variants and typos resolved)."""
    expected = [_clean_uri(d["doc_uri"]) for d in (expectations or {}).get("expected_retrieved_context", [])]
    if not expected:
        return None
    t = _tags(trace)
    found = {base_ref(r) for r in json.loads(t.get("returned_refs", "[]")) + json.loads(t.get("cited_refs", "[]"))}
    hit = [r for r in expected if base_ref(r) in found]
    return Feedback(value=round(len(hit) / len(expected), 3), rationale=f"expected: {expected} · found: {hit}")


@scorer
def reference_integrity(trace):
    """True when every cited document code exists (resolved typos and documents mentioned in the excerpts are accepted)."""
    t = _tags(trace)
    unverified = json.loads(t.get("unverified_refs", "[]"))
    notes = []
    if json.loads(t.get("approximate_refs", "[]")):
        notes.append(f"resolved: {json.loads(t['approximate_refs'])}")
    if json.loads(t.get("unindexed_refs", "[]")):
        notes.append(f"outside the corpus: {json.loads(t['unindexed_refs'])}")
    if unverified:
        notes.append(f"unverified: {unverified}")
    return Feedback(value=not unverified, rationale="; ".join(notes) or "all cited codes exist")


@scorer
def operations(outputs, trace):
    t = _tags(trace)
    ok = t.get("call_ok") == "true" or (t.get("call_ok") is None and bool(str(outputs or "").strip()))
    fb = [Feedback(name="call_ok", value=ok, rationale=t.get("error", ""))]
    if t.get("latency_s"):
        fb.append(Feedback(name="latency_s", value=float(t["latency_s"])))
    return fb


CODE_SCORERS = [document_recall, reference_integrity, operations]

# COMMAND ----------

# DBTITLE 1,Judge check — the judge model answers, otherwise fall back to the Databricks-managed judge model
JUDGE_MODEL = f"databricks:/{JUDGE_ENDPOINT}" if JUDGE_ENDPOINT else None
_sample = {"inputs": {"messages": [{"role": "user", "content": "What is the retention period of inspection records?"}]},
           "outputs": "According to QP-1457, inspection records are kept for 10 years.",
           "expectations": {"expected_facts": ["Inspection records are kept for 10 years."]}}


def _judges_work(model) -> bool:
    try:
        probe = build_llm_judges(model)
        probe[0](**_sample)          # Correctness
        probe[5](**_sample)          # fact_coverage
        return True
    except Exception as e:
        print(f"⚠️ judge model {model or 'Databricks-managed'} failed: {str(e)[:300]}")
        return False


if JUDGE_MODEL and not _judges_work(JUDGE_MODEL):
    print("   → falling back to the Databricks-managed judge model.")
    JUDGE_MODEL = None
LLM_JUDGES = build_llm_judges(JUDGE_MODEL)
SCORERS = LLM_JUDGES + CODE_SCORERS
print(f"Judge model: {JUDGE_MODEL or 'Databricks-managed'} · {len(LLM_JUDGES)} LLM judges · {len(CODE_SCORERS)} code scorers")

# Registration makes the LLM judges visible in the experiment's Judges tab. They are not started: no background cost.
for judge in LLM_JUDGES:
    try:
        judge.register(name=judge.name)
        print(f"  registered: {judge.name}")
    except Exception as e:
        msg = str(e).lower()
        print(f"  {'already registered' if 'exist' in msg else 'not registered (' + str(e)[:120] + ')'}: {judge.name}")

# COMMAND ----------

# DBTITLE 1,Case selection and evaluation run
def pick_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Diversified subset: picks in turn from each question type (stable for a given SEED)."""
    df = df.sample(frac=1, random_state=SEED)
    groups = [g for _, g in df.groupby(df["intent"].fillna("unknown"))]
    chosen, i = [], 0
    while len(chosen) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(chosen) < n:
                chosen.append(g.index[i])
        i += 1
    return df.loc[chosen]


if CASE_IDS:
    selection, subset_label = CASES[CASES["case_id"].isin(CASE_IDS)], f"cases:{','.join(CASE_IDS)}"
elif SAMPLE_N:
    selection, subset_label = pick_subset(CASES, SAMPLE_N), f"sample:{SAMPLE_N}:seed{SEED}"
else:
    selection, subset_label = CASES, "full"

n = len(selection)
print(f"Selection: {subset_label} → {n} cases × {len(ENDPOINTS)} endpoint(s) × {REPEATS} repetition(s)")
print(f"Estimated calls: {n * len(ENDPOINTS) * REPEATS} assistant calls, "
      f"~{n * len(ENDPOINTS) * REPEATS * len(LLM_JUDGES)} judge calls")
display(selection[["case_id", "intent", "difficulty", "final_answerability", "question"]])

RUN_IDS = []
if RUN_EVAL:
    # The full dataset is passed as the dataset object so the run is linked to it in the Datasets tab
    data = eval_ds if subset_label == "full" else selection[["inputs", "expectations"]].reset_index(drop=True)
    for endpoint in ENDPOINTS:
        for rep in range(1, REPEATS + 1):
            name = f"{endpoint} · {subset_label.split(':')[0]} · {time.strftime('%Y-%m-%d %H:%M')}" + (f" · r{rep}" if REPEATS > 1 else "")
            with mlflow.start_run(run_name=name) as run:
                mlflow.set_tags({"endpoint": endpoint, "subset": subset_label, "dataset": DATASET_NAME,
                                 "judge_model": JUDGE_MODEL or "databricks-managed"})
                mlflow.log_params({"n_cases": n, "repeat": rep, "excerpts_per_case": EXCERPTS_PER_CASE})
                mlflow.genai.evaluate(data=data, predict_fn=make_predict_fn(endpoint), scorers=SCORERS)
                RUN_IDS.append(run.info.run_id)
    print(f"✓ runs: {RUN_IDS}")
else:
    print("run_eval=false: nothing was evaluated.")

# COMMAND ----------

# DBTITLE 1,Report — scores with confidence intervals, breakdowns, failures (logged to the run)
REPORT_RUN_ID = ""   # empty = most recent run

VALUE_MAP = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "full": 1.0, "partial": 0.5, "none": 0.0,
             "no_contradiction": 1.0, "contradiction": 0.0, "correct_refusal": 1.0, "answered_anyway": 0.0,
             "not_applicable": None}
METRICS = {
    "correctness": "expected facts present (built-in)",
    "fact_coverage": "expected facts present: full=1, partial=0.5, none=0",
    "fact_contradiction": "no contradiction with the expected facts",
    "refusal_handling": "correct refusals when the answer is not in the documentation",
    "retrieval_groundedness": "answer supported by the cited documents' excerpts",
    "retrieval_sufficiency": "excerpts contain what the expected answer needs",
    "document_recall": "expected documents returned or cited",
    "reference_integrity": "every cited document code exists",
    "expectations_guidelines": "case-specific guidelines followed",
    "relevance_to_query": "answer addresses the question",
    "call_ok": "endpoint answered",
    "latency_s": "response time (median / p95)",
}
FAILURE_METRICS = ["correctness", "fact_contradiction", "refusal_handling", "retrieval_groundedness",
                   "reference_integrity", "expectations_guidelines", "call_ok"]


def to_number(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return VALUE_MAP.get(str(getattr(v, "value", v)).strip().lower())


def _assessment_value(a):
    for path in (("value",), ("feedback", "value")):
        obj = a
        try:
            for p in path:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None


def collect(run_id: str) -> pd.DataFrame:
    """One row per case: judge values, human labels (prefixed human::), rationales, answer."""
    rows = []
    for t in mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=run_id, return_type="list", max_results=2000):
        tags = t.info.tags or {}
        row, why = {"trace_id": t.info.trace_id, "case_id": tags.get("case_id"),
                    "answer": getattr(getattr(t, "data", None), "response", None) or getattr(t.info, "response_preview", "")}, {}
        for a in (getattr(t.info, "assessments", None) or []):
            if type(a).__name__ == "Expectation" or getattr(a, "expectation", None) is not None:
                continue
            value = to_number(_assessment_value(a))
            if value is None:
                continue
            source = str(getattr(getattr(a, "source", None), "source_type", "")).upper()
            key = f"human::{a.name}" if "HUMAN" in source else a.name
            row[key] = value
            if getattr(a, "rationale", None):
                why[key] = a.rationale
        row["_why"] = why
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.merge(CASES.drop(columns=["inputs", "expectations"], errors="ignore"), on="case_id", how="left") if len(df) else df


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, meaning in METRICS.items():
        if m not in df.columns or df[m].dropna().empty:
            continue
        s = df[m].dropna()
        if m == "latency_s":
            out.append({"metric": m, "meaning": meaning, "score": f"{s.median():.1f} s / {s.quantile(.95):.1f} s",
                        "95% CI": "", "n": len(s)})
            continue
        p = s.mean()
        if set(s.unique()) <= {0.0, 1.0}:
            lo, hi = wilson(p, len(s))
        else:
            half = 1.96 * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
            lo, hi = max(p - half, 0.0), min(p + half, 1.0)
        ci = f"{lo:.0%} – {hi:.0%}" if len(s) > 1 else ""
        out.append({"metric": m, "meaning": meaning, "score": f"{p:.0%}", "95% CI": ci, "n": len(s)})
    return pd.DataFrame(out)


def last_run_id():
    r = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["start_time DESC"], max_results=1)
    return r.iloc[0]["run_id"] if len(r) else None


run_id = REPORT_RUN_ID or (RUN_IDS[-1] if RUN_IDS else last_run_id())
if not run_id:
    print("No evaluation run yet: set run_eval=true.")
else:
    res = collect(run_id)
    run = mlflow.get_run(run_id)
    print(f"Run {run.info.run_name} ({run_id}) · {len(res)} cases")
    summary = summarize(res)
    display(summary)
    print("95% CI: range that most likely contains the true score; it is wide on few cases, which prevents over-reading.")

    cols = [c for c in ["correctness", "fact_coverage", "retrieval_groundedness", "retrieval_sufficiency", "document_recall"]
            if c in res.columns]
    for dim in ["intent", "final_answerability", "difficulty"]:
        if cols and res[dim].notna().any():
            print(f"\nBy {dim}:")
            display(res.groupby(res[dim].fillna("unknown"))[cols].agg(["mean", "count"]).round(2))

    fail_cols = [c for c in FAILURE_METRICS if c in res.columns]
    failures = res[(res[fail_cols] == 0).any(axis=1)].copy() if fail_cols else res.iloc[0:0].copy()
    if len(failures):
        failures["failed"] = failures.apply(lambda r: ", ".join(c for c in fail_cols if r.get(c) == 0), axis=1)
        failures["rationales"] = failures.apply(
            lambda r: " | ".join(f"{c}: {r['_why'].get(c, '')}" for c in fail_cols if r.get(c) == 0), axis=1)
        print(f"\n{len(failures)} failing case(s):")
        display(failures[["case_id", "intent", "question", "failed", "rationales", "trace_id"]])

    with mlflow.start_run(run_id=run_id):
        for m in METRICS:
            if m in res.columns and res[m].notna().any():
                mlflow.log_metric(f"report/{m}", float(res[m].mean()))
        mlflow.log_table(summary, "report/summary.json")
        if len(failures):
            mlflow.log_table(failures.drop(columns=["_why"]).astype(str), "report/failures.json")
    print("✓ report logged to the run (Metrics: report/*, Artifacts: report/)")

# COMMAND ----------

# DBTITLE 1,Run comparison — same subset only
runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["start_time DESC"], max_results=20)
metric_cols = [c for c in runs.columns if c.startswith("metrics.report/")]
if metric_cols:
    view = runs[["tags.mlflow.runName", "tags.subset", "params.n_cases", "start_time"] + metric_cols].copy()
    view.columns = ["run", "subset", "n", "date"] + [c.replace("metrics.report/", "") for c in metric_cols]
    display(view.round(3))
    print("Compare runs evaluated on the same subset only. On 25-30 cases, differences below ~10 points are noise.")
else:
    print("No run with a report yet.")

# COMMAND ----------

# DBTITLE 1,Human labels — rate a few answers; agreement with the judges tells whether the scores can be trusted
# Each rating is stored in MLflow as a HUMAN assessment named "fact_coverage" on the trace, next to the judge's own
# "fact_coverage" verdict. The same labels feed the judge alignment below. Agreement ≥ 85% → the judge is reliable.
import html as html_mod

import ipywidgets as widgets
from IPython.display import display as ipy_display
from mlflow.entities import AssessmentSource, AssessmentSourceType

LABEL_RUN_ID = ""   # empty = most recent run
label_run = LABEL_RUN_ID or (RUN_IDS[-1] if RUN_IDS else last_run_id())
ME = spark.sql("SELECT current_user()").first()[0]
HUMAN = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=ME)
EXPECTATIONS_BY_CASE = dict(zip(CASES["case_id"], CASES["expectations"]))


def agreement_report(df: pd.DataFrame) -> str:
    if "human::fact_coverage" not in df.columns or "fact_coverage" not in df.columns:
        return "<i>No human rating yet.</i>"
    both = df.dropna(subset=["human::fact_coverage", "fact_coverage"])
    if not len(both):
        return "<i>No case rated by both you and the judge yet.</i>"
    exact = (both["human::fact_coverage"] == both["fact_coverage"]).mean()
    binary = ((both["human::fact_coverage"] == 1.0) == (both["fact_coverage"] == 1.0)).mean()
    verdict = "reliable" if binary >= 0.85 else "needs alignment (see next cell)"
    return (f"<b>fact_coverage</b>: exact agreement {exact:.0%}, full-vs-not-full agreement {binary:.0%} "
            f"on {len(both)} rated case(s) → {verdict}")


class HumanLabeler:
    OPTIONS = [("✅ Full", "full"), ("🟡 Partial", "partial"), ("❌ None", "none")]

    def __init__(self, df: pd.DataFrame):
        self.df, self.i = df.reset_index(drop=True), 0
        self.card, self.stats = widgets.HTML(), widgets.HTML()
        buttons = [widgets.Button(description=label) for label, _ in self.OPTIONS]
        for b, (_, value) in zip(buttons, self.OPTIONS):
            b.on_click(lambda _, v=value: self._rate(v))
        skip = widgets.Button(description="Skip")
        skip.on_click(lambda _: self._next())
        self.box = widgets.VBox([self.card, widgets.HBox(buttons + [skip]), self.stats])
        self._render()

    def _render(self):
        if self.i >= len(self.df):
            self.card.value = "<b>All selected cases are rated.</b>"
        else:
            r = self.df.iloc[self.i]
            exp = EXPECTATIONS_BY_CASE.get(r["case_id"], {}) or {}
            expected = "<br>".join("• " + html_mod.escape(f) for f in exp.get("expected_facts", [])) \
                or html_mod.escape(str(exp.get("expected_response", "")))
            judge = r.get("fact_coverage")
            self.card.value = (
                f"<div style='font-family:sans-serif;font-size:14px;border:1px solid #ddd;border-radius:8px;padding:12px'>"
                f"<b>Case {self.i + 1}/{len(self.df)}</b> · #{r['case_id']} · judge fact_coverage = {judge}<br><br>"
                f"<b>Question</b><br>{html_mod.escape(str(r.get('question')))}<br><br>"
                f"<b>Expected</b><br>{expected}<br><br><b>Assistant answer</b>"
                f"<div style='max-height:320px;overflow:auto;background:#fafafa;padding:8px'>"
                f"{html_mod.escape(str(r.get('answer'))).replace(chr(10), '<br>')}</div></div>")
        self.stats.value = agreement_report(collect(label_run))

    def _rate(self, value):
        r = self.df.iloc[self.i]
        mlflow.log_feedback(trace_id=r["trace_id"], name="fact_coverage", value=value, source=HUMAN)
        self._next()

    def _next(self):
        self.i += 1
        self._render()


if label_run:
    labels = collect(label_run)
    todo = labels[labels["human::fact_coverage"].isna()] if "human::fact_coverage" in labels.columns else labels
    print(f"{len(todo)} case(s) to rate on run {label_run}")
    ipy_display(HumanLabeler(todo).box)
else:
    print("No evaluation run yet.")

# COMMAND ----------

# DBTITLE 1,Judge alignment — adapts fact_coverage to your ratings (MemAlign), registered as fact_coverage_aligned
RUN_ALIGNMENT = False   # needs at least 10 cases rated in the cell above (50+ gives better results)

if RUN_ALIGNMENT:
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    rated = [t for t in mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=label_run, return_type="list")
             if any(a.name == "fact_coverage" and "HUMAN" in str(a.source.source_type).upper()
                    for a in (t.info.assessments or []))]
    print(f"{len(rated)} rated trace(s)")
    if len(rated) >= 10:
        base_judge = next(j for j in LLM_JUDGES if j.name == "fact_coverage")
        optimizer = MemAlignOptimizer(model=JUDGE_MODEL) if JUDGE_MODEL else MemAlignOptimizer()
        aligned = base_judge.align(traces=rated, optimizer=optimizer)
        aligned.register(name="fact_coverage_aligned")
        print("✓ fact_coverage_aligned registered: use it in place of fact_coverage in build_llm_judges once validated.")
    else:
        print("Rate at least 10 cases first.")

# COMMAND ----------

# DBTITLE 1,Review export — one Markdown file per run with every case, answer, verdict and rationale
EXPORT_RUN_ID = ""      # empty = most recent run
MAX_ANSWER_CHARS = 3000
FAILED_ONLY = False

rid = EXPORT_RUN_ID or (RUN_IDS[-1] if RUN_IDS else last_run_id())
if rid:
    run = mlflow.get_run(rid)
    res = collect(rid)
    if FAILED_ONLY:
        res = res[(res[[c for c in FAILURE_METRICS if c in res.columns]] == 0).any(axis=1)]
    traces = {t.info.trace_id: t for t in
              mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=rid, return_type="list", max_results=2000)}
    exp_by_case = dict(zip(CASES["case_id"], CASES["expectations"]))
    summary = summarize(res)
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
    out = [f"# Knowledge Assistant evaluation — {run.info.run_name}", f"- run_id: `{rid}`",
           f"- subset: {run.data.tags.get('subset', '?')} · {len(res)} cases · judge model: {run.data.tags.get('judge_model', '?')}",
           "", "## Summary", "", "| " + " | ".join(summary.columns) + " |", "|" + "---|" * len(summary.columns)]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in summary.itertuples(index=False)]
    for _, r in res.sort_values("case_id").iterrows():
        exp = exp_by_case.get(r["case_id"], {}) or {}
        tags = traces[r["trace_id"]].info.tags if r["trace_id"] in traces else {}
        answer = str(r.get("answer") or "")
        out += ["", "---", "", f"### #{r['case_id']} · {r.get('intent')} · {r.get('difficulty')} · "
                f"answer in documentation: {r.get('final_answerability')}",
                f"- trace: `{r['trace_id']}` · latency: {tags.get('latency_s', '?')} s", "", f"**Question**: {r.get('question')}", ""]
        out += (["**Expected facts**:"] + [f"- {f}" for f in exp["expected_facts"]]) if exp.get("expected_facts") \
            else [f"**Expected answer**: {exp.get('expected_response', '')}"]
        if exp.get("guidelines"):
            out += ["", "**Guidelines**:"] + [f"- {g}" for g in exp["guidelines"]]
        out += ["", f"**Expected documents**: {', '.join(_clean_uri(d['doc_uri']) for d in exp.get('expected_retrieved_context', [])) or '—'}",
                f"**Documents returned by the assistant**: {', '.join(json.loads(tags.get('returned_refs', '[]'))) or '—'}",
                f"**Documents cited in the answer**: {', '.join(json.loads(tags.get('cited_refs', '[]'))) or '—'}",
                "", "**Assistant answer**:", "", "> " + answer[:MAX_ANSWER_CHARS].replace("\n", "\n> ")
                + (f"\n> […] ({len(answer)} characters)" if len(answer) > MAX_ANSWER_CHARS else ""), "", "**Scores**:"]
        for m in METRICS:
            if m in r and pd.notna(r[m]):
                why = (r["_why"] or {}).get(m, "")
                out.append(f"- {m} = {r[m]:.2f}" + (f" — {why}" if why else ""))
        if pd.notna(r.get("human::fact_coverage")):
            out.append(f"- **human fact_coverage** = {r['human::fact_coverage']:.1f}")
    path = f"/Workspace/Users/{ME}/qualibot_evaluation_{rid[:8]}.md"
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"✓ {path} ({sum(len(l) for l in out):,} characters) — Workspace browser → right-click → Download")
else:
    print("No evaluation run yet.")
