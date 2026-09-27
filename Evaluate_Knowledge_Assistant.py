# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Evaluation
# MAGIC
# MAGIC Evaluates a Knowledge Assistant endpoint against the golden evaluation dataset stored in Unity Catalog, with
# MAGIC MLflow GenAI evaluation, and writes the results to Unity Catalog tables for the dashboard.
# MAGIC
# MAGIC ### Quality dimensions
# MAGIC | Axis | Scorer | Type | Question answered |
# MAGIC |---|---|---|---|
# MAGIC | Correctness | `correctness` | built-in | Does the answer contain the expected facts (or match the expected answer)? |
# MAGIC | | `fact_coverage` | LLM judge | full / partial / none — finer than a yes/no; can be aligned on human ratings |
# MAGIC | | `fact_contradiction` | LLM judge | Does the answer contradict an expected fact? |
# MAGIC | | `refusal_handling` | LLM judge | When the expected behaviour is a refusal, does the assistant refuse without inventing? |
# MAGIC | Answer | `relevance` ¹, `language_match` ¹ | LLM judge | Does the answer address the question, in the user's language? |
# MAGIC | | `expectations_guidelines` | built-in | Case-specific guidelines (mandatory citation, …) |
# MAGIC | Faithfulness | `groundedness` ¹ | LLM judge on the `RETRIEVER` step | Is the answer supported by the excerpts of the documents it cites? |
# MAGIC | | `missed_answer` ¹ | LLM judge on the `RETRIEVER` step | Does the answer say "not found" while the excerpts contain the information? |
# MAGIC | Retrieval | `retrieval_sufficiency` | built-in, on the `RETRIEVER` step | Do those excerpts contain what the expected answer needs? (cases with expected facts) |
# MAGIC | | `document_recall` | code | Share of the expected documents the assistant returned or cited |
# MAGIC | | `reference_integrity` ¹ | code | Every cited document code exists (typos such as missing zero padding are resolved, not penalised) |
# MAGIC | Operations | `call_ok`, `latency_s` | code | Endpoint availability and response time |
# MAGIC
# MAGIC ¹ identical in the production monitoring notebook, so that golden-dataset and production results can be compared.
# MAGIC
# MAGIC ### Outputs
# MAGIC | Where | Content |
# MAGIC |---|---|
# MAGIC | `ka_eval_runs` | one row per evaluation run: endpoint, subset, judge model, scorer configuration |
# MAGIC | `ka_eval_metrics` | one row per run × metric: score, 95% confidence interval, number of cases |
# MAGIC | `ka_eval_results` | one row per run × case: question, answer, documents, one column per metric, human rating |
# MAGIC | `ka_eval_assessments` | one row per run × case × scorer (value, numeric value, rationale, error), human ratings included |
# MAGIC | `v_ka_eval_metrics`, `v_ka_eval_case_history`, `v_quality_shared_scorers` | dashboard-ready views (the last one puts the shared scorers of evaluation and production side by side) |
# MAGIC | MLflow experiment | **Runs**: one per endpoint and repetition · **Traces**: one per case (assistant call, cited excerpts, every verdict) · **Judges / Scorers**: every scorer above, registered but not scheduled · **Datasets**: the golden dataset, linked to every run |
# MAGIC
# MAGIC ### How to use
# MAGIC 1. Run the cells down to **Case selection** with `run_eval=false` to review the selected cases and the estimated number of calls.
# MAGIC 2. Set `run_eval=true` (start with `sample_n=5`), then read the **Report**; the tables are written by the report.
# MAGIC 3. Use **Human labels** to rate a few answers: the agreement between you and the judges tells whether the scores can
# MAGIC    be trusted, and the same labels can align `fact_coverage` (**Judge alignment**).

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

NEEDED = {"mlflow": (3, 11)}  # typed make_judge feedback; databricks:/ judge models without LiteLLM


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
dbutils.widgets.text("output_schema", "uat_proj.qualibot")         # result tables and views

DATASET_NAME = dbutils.widgets.get("dataset_name").strip()
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
ENDPOINTS = [e.strip() for e in dbutils.widgets.get("endpoints").split(",") if e.strip()]
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id").strip()
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint").strip()
SAMPLE_N = int(dbutils.widgets.get("sample_n")) if dbutils.widgets.get("sample_n").strip() else None
CASE_IDS = [c.strip() for c in dbutils.widgets.get("case_ids").split(",") if c.strip()]
REPEATS = max(1, int(dbutils.widgets.get("repeats") or 1))
RUN_EVAL = dbutils.widgets.get("run_eval") == "true"
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()

RUNS_TABLE = f"{OUTPUT_SCHEMA}.ka_eval_runs"
METRICS_TABLE = f"{OUTPUT_SCHEMA}.ka_eval_metrics"
RESULTS_TABLE = f"{OUTPUT_SCHEMA}.ka_eval_results"
ASSESSMENTS_TABLE = f"{OUTPUT_SCHEMA}.ka_eval_assessments"
PRODUCTION_ASSESSMENTS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_assessments"   # written by the production scoring job

# Retrieval context used to verify the answers (documents cited by the assistant)
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None          # source table of the index; None = read from the index definition
EXCERPTS_PER_CASE = 10
EXCERPT_MAX_CHARS = 1500

# Optional case metadata (question type, difficulty, expected answerability) written by the dataset builder
GOLDEN_CACHE_TABLE = "uat_landingzone.qualibot.qualibot_eval_cache"

MAX_PARALLEL_CALLS = 3           # capacity limit of the Knowledge Assistant endpoints
SEED = 42
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX", "BG", "RO", "PL", "TN"]
TRACES_CATALOG, TRACES_SCHEMA = "uat_proj", "qualibot"      # Unity Catalog location of the evaluation traces

# COMMAND ----------

# DBTITLE 1,Shared scorers and helpers — identical in Evaluate_Knowledge_Assistant.py and Score_Production_QA.py
# This cell is identical in both notebooks (checked by tests/test_shared.py): the same judges score the golden dataset
# and the production turns, so that their results can be compared in the dashboard.
import hashlib
import json
import re
from typing import Literal

import mlflow
from mlflow import MlflowClient
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import delete_scorer, scorer
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, LongType, StringType

_TEXT_FRAGMENT = re.compile(r"#:~:text=[^)\s\]>]*")


def clean_answer(text) -> str:
    """Answer as read by the judges: the text fragments of citation links (#:~:text=…, often longer than the cited
    passage itself) are removed; the links and the passages quoted in the footnotes are kept."""
    return _TEXT_FRAGMENT.sub("", str(text or ""))


CONTEXT = """Qualibot is an internal assistant answering questions about the QUALITY documentation of an aerospace
manufacturer (procedures, work instructions, forms, templates, quality rules) for two divisions: AS (Aerostructures)
and IS (Interconnection Systems). Documents are identified by codes such as PRLAT508, QP-1457, NF-10065, INAQ619_FR.
It must answer only from those documents, cite them, answer in the user's language, and say so when the information is
not in the documentation.

{{ inputs }} holds `messages`, the conversation as the assistant saw it (oldest first; the last user message is the
question). {{ outputs }} is the assistant answer under evaluation. Write the rationale in English, in one or two
sentences.
"""

SHARED_JUDGES = {
    "relevance": (
        "The answer addresses what the user asked, given the whole conversation.",
        CONTEXT + """
Does the answer address what the user asked, given the whole conversation? When the last user message asks to modify,
correct or restate the previous answer ("remove document X", "shorter", "same for Y"), judge whether the answer applies
that request. An appropriate refusal of an out-of-scope request, or a justified "not found", counts as relevant.
Return yes or no."""),
    "language_match": (
        "The answer is written in the language of the user's last message.",
        CONTEXT + """
Is the answer written in the language of the last user message? Document titles and codes in another language do not
count. Return yes or no."""),
}


def shared_llm_judges(model) -> list:
    kw = {"model": model} if model else {}
    return [make_judge(name=name, description=desc, instructions=instructions, feedback_value_type=Literal["yes", "no"], **kw)
            for name, (desc, instructions) in SHARED_JUDGES.items()]


@scorer(name="groundedness", description="The answer's claims are supported by the excerpts of the documents it cites "
                                         "(supported / partially_supported / not_supported).")
def groundedness(inputs, outputs, trace):
    """Checks the answer's factual claims against the excerpts of the documents it cites (RETRIEVER step).
    Not applicable (no assessment) when the answer cites no document found in the index."""
    from typing import Literal

    from mlflow.entities import SpanType
    from mlflow.genai.judges import make_judge

    docs = [d if isinstance(d, dict) else d.to_dict()
            for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    if not docs:
        return None
    excerpts = "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri')}] {d.get('page_content')}" for d in docs)
    question = next((m["content"] for m in reversed(inputs["messages"]) if m["role"] == "user"), "")
    judge = make_judge(
        name="groundedness",
        instructions="""You verify an answer of an assistant on aerospace quality documentation. {{ inputs }} holds the
user's question and EXCERPTS of the documents the answer cites; {{ outputs }} is the answer.
List mentally the answer's key factual claims (values, thresholds, deadlines, roles, steps, document identities,
definitions), ignoring greetings, generic advice and questions to the user. The excerpts are only a SUBSET of the
documents: a claim absent from the excerpts is not verifiable, which is NOT a contradiction.
Return:
- supported: every verifiable claim is stated by an excerpt;
- partially_supported: some claims are only partly supported or close but not exact;
- not_supported: at least one claim is contradicted by the excerpts, or the excerpts of that document clearly show it
  does not say this.
Write the rationale in English and name the unsupported claims, if any.""",
        feedback_value_type=Literal["supported", "partially_supported", "not_supported"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"question": question, "excerpts": excerpts}, outputs=outputs)


@scorer(name="missed_answer", description="yes when the answer says the information is not available while the excerpts "
                                          "of the cited documents contain it.")
def missed_answer(inputs, outputs, trace):
    """yes when the answer says the information is not available (or leaves a part unanswered) while the excerpts of
    the cited documents contain it. Not applicable (no assessment) when no excerpt was retrieved."""
    from typing import Literal

    from mlflow.entities import SpanType
    from mlflow.genai.judges import make_judge

    docs = [d if isinstance(d, dict) else d.to_dict()
            for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    if not docs:
        return None
    excerpts = "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri')}] {d.get('page_content')}" for d in docs)
    question = next((m["content"] for m in reversed(inputs["messages"]) if m["role"] == "user"), "")
    judge = make_judge(
        name="missed_answer",
        instructions="""{{ inputs }} holds a user's question and excerpts of quality documents; {{ outputs }} is an
assistant's answer. Return yes if the answer says the information is not available, or leaves a part of the question
unanswered, while the excerpts DO contain that information; otherwise return no. In the rationale (English), state what
was missed, if anything.""",
        feedback_value_type=Literal["yes", "no"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"question": question, "excerpts": excerpts}, outputs=outputs)


@scorer(name="reference_integrity", description="Every document code cited in the answer exists (resolved typos and "
                                                "documents outside the corpus are accepted).")
def reference_integrity(trace):
    """True when every cited document code exists: resolved typos (zero padding) and documents outside the corpus
    mentioned in the excerpts are accepted; codes found nowhere are reported as unverified."""
    import json

    from mlflow.entities import Feedback

    tags = trace.info.tags or {}
    unverified = json.loads(tags.get("unverified_refs", "[]"))
    notes = []
    if json.loads(tags.get("approximate_refs", "[]")):
        notes.append(f"resolved: {json.loads(tags['approximate_refs'])}")
    if json.loads(tags.get("unindexed_refs", "[]")):
        notes.append(f"outside the corpus: {json.loads(tags['unindexed_refs'])}")
    if unverified:
        notes.append(f"unverified: {unverified}")
    return Feedback(value=not unverified, rationale="; ".join(notes) or "all cited codes exist")


SHARED_TRACE_SCORERS = [groundedness, missed_answer, reference_integrity]

# ── Numeric form of a verdict: 1 = pass, 0 = fail, 0.5 = partial; counts and durations as is; NULL for labels ──
SCORE_VALUES = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "full": 1.0, "partial": 0.5, "none": 0.0,
                "supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0,
                "no_contradiction": 1.0, "contradiction": 0.0, "correct_refusal": 1.0, "answered_anyway": 0.0,
                "good": 1.0, "acceptable": 0.5, "bad": 0.0, "up": 1.0, "down": 0.0}
LABEL_SCORERS = {"question_intent", "answer_type", "user_reaction"}   # categorical: no numeric form
INVERTED_SCORERS = {"missed_answer"}                                                   # "yes" is the failure


def numeric_value(name: str, value):
    if value is None or name in LABEL_SCORERS:
        return None
    if isinstance(value, bool):
        x = float(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        x = SCORE_VALUES.get(str(getattr(value, "value", value)).strip().lower())
    return None if x is None else (1.0 - x if name in INVERTED_SCORERS else x)


def assessment_row(a) -> dict:
    """Name, raw value, rationale, source type and error of an MLflow assessment."""
    value = getattr(a, "value", None)
    fb = getattr(a, "feedback", None)
    err = getattr(fb, "error", None) if fb is not None else None
    return {"name": a.name, "value": getattr(value, "value", value), "rationale": getattr(a, "rationale", None),
            "source_type": str(getattr(getattr(a, "source", None), "source_type", "") or "").split(".")[-1].upper(),
            "error": (getattr(err, "error_message", None) or str(err))[:1000] if err else None}


# ── Scorer registration (Judges / Scorers tab), only when the scorer definitions changed ──
_SCORERS_TAG = "qualibot.scorers_config_id"


def scorer_definition(s) -> str:
    if getattr(s, "instructions", None):
        return f"{s.name}|{s.instructions}|{getattr(s, 'feedback_value_type', '')}"
    if getattr(s, "_original_func", None) is not None:
        return f"{s.name}|{s.model_dump().get('call_source')}"
    return f"{s.name}|{type(s).__name__}"


def scorers_config_id(scorers, *extra) -> str:
    """Fingerprint of the scorer definitions (instructions, code) and of any extra setting (model, rules)."""
    payload = json.dumps([scorer_definition(s) for s in scorers] + [str(x) for x in extra])
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def publish_scorers(scorers, experiment_id: str, config_id: str):
    """Registers every scorer in the experiment, replacing a registered scorer of the same name. Registration only:
    nothing is scheduled, so there is no background cost. Skipped when this configuration is already registered."""
    client = MlflowClient()
    if client.get_experiment(experiment_id).tags.get(_SCORERS_TAG) == config_id:
        print(f"Scorers already registered (configuration {config_id}).")
        return
    failed = []
    for s in scorers:
        try:
            try:
                delete_scorer(name=s.name, experiment_id=experiment_id)
            except Exception:
                pass                          # not registered yet
            s.register(name=s.name, experiment_id=experiment_id)
            print(f"  registered: {s.name}")
        except Exception as e:
            failed.append(s.name)
            print(f"  not registered: {s.name} ({str(e)[:150]})")
    if not failed:
        client.set_experiment_tag(experiment_id, _SCORERS_TAG, config_id)


# ── Unity Catalog tables: documented DDL, row replacement by key ──
def _sql_text(text) -> str:
    return str(text or "").replace("\\", "\\\\").replace("'", "\\'")


def ensure_table(name: str, schema, comment: str, column_docs: dict):
    """Creates the table with its table and column comments, or adds the columns it lacks."""
    col = lambda f: f"`{f.name}` {f.dataType.simpleString().upper()} COMMENT '{_sql_text(column_docs.get(f.name, ''))}'"
    if not spark.catalog.tableExists(name):
        spark.sql(f"CREATE TABLE {name} ({', '.join(col(f) for f in schema.fields)}) COMMENT '{_sql_text(comment)}'")
        return
    existing = set(spark.table(name).columns)
    missing = [f for f in schema.fields if f.name not in existing]
    if missing:
        spark.sql(f"ALTER TABLE {name} ADD COLUMNS ({', '.join(col(f) for f in missing)})")


def to_cell(v, field):
    """Python value → value accepted by the Spark field type (NaN/None, numpy scalars and arrays, casts)."""
    if hasattr(v, "item") and not isinstance(v, (str, bytes, list, dict)) and getattr(v, "ndim", 0) == 0:
        v = v.item()
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if v is None or (isinstance(v, float) and v != v):
        return [] if isinstance(field.dataType, ArrayType) else None
    t = field.dataType
    if isinstance(t, ArrayType):
        return [str(x) for x in v]
    if isinstance(t, BooleanType):
        return bool(v)
    if isinstance(t, LongType):
        return int(v)
    if isinstance(t, DoubleType):
        return float(v)
    if isinstance(t, StringType):
        return str(v)
    return v


def replace_rows(table: str, schema, rows: list, keys: list):
    """Writes rows into the table: existing rows sharing their key values are replaced (idempotent re-runs)."""
    if not rows:
        return
    view = f"_rows_{table.split('.')[-1]}"
    spark.createDataFrame([tuple(to_cell(r.get(f.name), f) for f in schema.fields) for r in rows], schema) \
         .createOrReplaceTempView(view)
    on = " AND ".join(f"t.`{k}` = s.`{k}`" for k in keys)
    spark.sql(f"MERGE INTO {table} t USING (SELECT DISTINCT {', '.join(keys)} FROM {view}) s ON {on} WHEN MATCHED THEN DELETE")
    cols = ", ".join(f"`{f.name}`" for f in schema.fields)
    spark.sql(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {view}")

# COMMAND ----------

# DBTITLE 1,Connections — experiment, dataset, cases
import math
import os
import time

import mlflow.genai.datasets as gdatasets
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import display
from mlflow.entities.trace_location import UnityCatalog

w = WorkspaceClient()
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID          # traces are stored in Unity Catalog
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(MAX_PARALLEL_CALLS)
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"   # no extra assistant call before the run

# The evaluation experiment stores its traces in Unity Catalog tables <prefix>_otel_*; created once, reused afterwards
if mlflow.get_experiment_by_name(EXPERIMENT_PATH) is None:
    w.workspace.mkdirs(EXPERIMENT_PATH.rsplit("/", 1)[0])
    mlflow.set_experiment(experiment_name=EXPERIMENT_PATH, trace_location=UnityCatalog(
        catalog_name=TRACES_CATALOG, schema_name=TRACES_SCHEMA, table_prefix=EXPERIMENT_PATH.rsplit("/", 1)[1]))
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
_LANG = re.compile(r"[-_. ](%s)$" % "|".join(LANG_SUFFIXES), re.I)
_CODE = re.compile(r"^(?=.*\d)(?=(?:.*[A-Z]){2})[A-Z][A-Z0-9_.\-]{2,28}( (%s))?$" % "|".join(LANG_SUFFIXES))
_BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def _strip(s) -> str:
    return _LANG.sub("", _EXT.sub("", str(s).strip().split("/")[-1]))


def base_ref(s) -> str:
    """Document key: PRLAT538_FR, prlat-538 GB and PRLAT538.FR → PRLAT538; IN_APO_006 (typo present in some documents)
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


def clean_ref(code) -> str:
    """Document code without the "REF:" prefix some expectations carry."""
    return re.sub(r"^\s*REF\s*:\s*", "", str(code)).strip()


def resolve_refs(refs) -> list:
    """Cited codes → exact REF values of the index (all language variants)."""
    return sorted({real for r in refs if len(base_ref(r)) >= 4 for real in REFS_BY_BASE.get(base_ref(r), set())})


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
    excerpt_keys = {base_ref(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{2,28}", excerpt_text or "")}
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
        tags = {"case_id": str(CASE_BY_QUESTION.get(question, "")), "endpoint": endpoint, "judge_model": JUDGE_MODEL or ""}
        t0 = time.time()
        try:
            raw = call_assistant(endpoint, messages)
        except Exception as e:
            mlflow.update_current_trace(tags={**tags, "call_ok": "false", "error": str(e)[:500]})
            cited_document_excerpts(question, [])
            return ""
        raw_answer = _extract_text(raw) or ""
        answer = clean_answer(raw_answer)          # as read by the judges; the raw answer stays in knowledge_assistant
        returned = sorted(refs_from_response(raw))
        cited = sorted(code_like(raw_answer))
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

# DBTITLE 1,Evaluation scorers — reference-based judges (expected facts, guidelines, documents) and operations
from mlflow.genai.scorers import Correctness, ExpectationsGuidelines

EVALUATION_JUDGES = {
    "fact_coverage": (
        "Share of the expected facts present in the answer: full / partial / none (not_applicable for refusal cases).",
        Literal["full", "partial", "none", "not_applicable"],
        """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Compare the answer in {{ outputs }} to the question in {{ inputs }} and to the expected facts in
{{ expectations }}. Be strict on substance and tolerant on wording (paraphrases, synonyms, other language).
Return:
- full: every expected fact is present;
- partial: some expected facts are present, others are missing or imprecise;
- none: no expected fact is present;
- not_applicable: the expectations contain no expected_facts (the expected behaviour is a refusal)."""),
    "fact_contradiction": (
        "The answer states nothing incompatible with the expected facts (a different value, deadline, role or document).",
        Literal["no_contradiction", "contradiction", "not_applicable"],
        """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Check whether the answer in {{ outputs }} states anything incompatible with the expected facts in
{{ expectations }} for the question in {{ inputs }}: a different value, threshold, deadline, responsible role or document.
Missing facts are NOT contradictions. Return no_contradiction, contradiction, or not_applicable when the expectations
contain no expected_facts."""),
    "refusal_handling": (
        "When the expected behaviour is a refusal, the assistant refuses without fabricating an answer.",
        Literal["correct_refusal", "answered_anyway", "not_applicable"],
        """You evaluate an assistant answering questions about the quality documentation of an aerospace
manufacturer. Apply this judge only when the expectations in {{ expectations }} contain no expected_facts: the expected
behaviour is then to state that the information is not in the documentation, or to decline an out-of-scope request.
For the question in {{ inputs }} and the answer in {{ outputs }}, return:
- correct_refusal: the answer clearly says the information is not available (or declines) and does not fabricate an
  answer; mentioning what was found nearby or asking for clarification is acceptable;
- answered_anyway: the answer fabricates an answer to the question;
- not_applicable: the expectations contain expected_facts."""),
}


@scorer(name="retrieval_sufficiency", description="The excerpts of the cited documents contain what the expected facts "
                                                  "need (cases with expected facts only).")
def retrieval_sufficiency(expectations, trace):
    """Built-in sufficiency judge on the cited documents' excerpts, for cases with expected facts only (a refusal case
    has nothing to retrieve)."""
    from mlflow.genai.scorers import RetrievalSufficiency

    if not (expectations or {}).get("expected_facts"):
        return None
    return RetrievalSufficiency(model=trace.info.tags.get("judge_model") or None)(
        trace=trace, expectations={"expected_facts": expectations["expected_facts"]})


@scorer(name="document_recall", description="Share of the expected documents returned or cited by the assistant.")
def document_recall(expectations, trace):
    """Share of the expected documents returned or cited by the assistant. Codes are compared with a key insensitive to
    language suffix, separators, case and zero padding (IN_APO_006 = IN_APO_0006, PRLAT549.FR = PRLAT549_GB)."""
    import json
    import re

    from mlflow.entities import Feedback

    def key(code):
        code = re.sub(r"^\s*REF\s*:\s*", "", str(code)).strip().split("/")[-1]
        code = re.sub(r"\.(pdf|docx?|xlsx?|pptx?|txt)$", "", code, flags=re.I)
        code = re.sub(r"[-_. ](FR|GB|EN|UK|CZ|ES|DE|PT|IT|MX|BG|RO|PL|TN)$", "", code, flags=re.I)
        return "".join(str(int(g)) if g.isdigit() else g for g in re.findall(r"[A-Za-z]+|\d+", code.upper()))

    expected = [str(d["doc_uri"]) for d in (expectations or {}).get("expected_retrieved_context", [])]
    if not expected:
        return None
    tags = trace.info.tags or {}
    found = {key(r) for r in json.loads(tags.get("returned_refs", "[]")) + json.loads(tags.get("cited_refs", "[]"))}
    hit = [r for r in expected if key(r) in found]
    return Feedback(value=round(len(hit) / len(expected), 3), rationale=f"expected: {expected} · found: {hit}")


@scorer(name="operations", description="call_ok: the endpoint answered · latency_s: response time of the assistant.")
def operations(outputs, trace):
    """call_ok: the endpoint answered · latency_s: response time of the assistant."""
    from mlflow.entities import Feedback

    tags = trace.info.tags or {}
    ok = tags.get("call_ok") == "true" or (tags.get("call_ok") is None and bool(str(outputs or "").strip()))
    feedbacks = [Feedback(name="call_ok", value=ok, rationale=tags.get("error") or None)]
    if tags.get("latency_s"):
        feedbacks.append(Feedback(name="latency_s", value=float(tags["latency_s"])))
    return feedbacks


def build_scorers(model):
    """(LLM judges called for every case, trace judges called when applicable, code scorers)."""
    kw = {"model": model} if model else {}
    every_case = [Correctness(**kw), ExpectationsGuidelines(**kw)] + \
                 [make_judge(name=n, description=d, feedback_value_type=t, instructions=i, **kw)
                  for n, (d, t, i) in EVALUATION_JUDGES.items()] + shared_llm_judges(model)
    return (every_case, [groundedness, missed_answer, retrieval_sufficiency],
            [reference_integrity, document_recall, operations])

# COMMAND ----------

# DBTITLE 1,Judge check and registration — the judge model answers, otherwise fall back to the Databricks-managed judge
JUDGE_MODEL = f"databricks:/{JUDGE_ENDPOINT}" if JUDGE_ENDPOINT else None
_sample = {"inputs": {"messages": [{"role": "user", "content": "What is the retention period of inspection records?"}]},
           "outputs": "According to QP-1457, inspection records are kept for 10 years.",
           "expectations": {"expected_facts": ["Inspection records are kept for 10 years."]}}


def _judges_work(model) -> bool:
    try:
        every_case = build_scorers(model)[0]
        every_case[0](**_sample)                                            # built-in judge (correctness)
        next(j for j in every_case if j.name == "fact_coverage")(**_sample)  # custom judge
        return True
    except Exception as e:
        print(f"⚠️ judge model {model or 'Databricks-managed'} failed: {str(e)[:300]}")
        return False


if JUDGE_MODEL and not _judges_work(JUDGE_MODEL):
    print("   → falling back to the Databricks-managed judge model.")
    JUDGE_MODEL = None
LLM_JUDGES, TRACE_JUDGES, CODE_SCORERS = build_scorers(JUDGE_MODEL)
SCORERS = LLM_JUDGES + TRACE_JUDGES + CODE_SCORERS
SCORERS_CONFIG_ID = scorers_config_id(SCORERS, JUDGE_MODEL, EXCERPTS_PER_CASE)
print(f"Judge model: {JUDGE_MODEL or 'Databricks-managed'} · {len(LLM_JUDGES) + len(TRACE_JUDGES)} LLM judges · "
      f"{len(CODE_SCORERS)} code scorers · configuration {SCORERS_CONFIG_ID}")
publish_scorers(SCORERS, EXPERIMENT_ID, SCORERS_CONFIG_ID)

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
      f"~{n * len(ENDPOINTS) * REPEATS * (len(LLM_JUDGES) + len(TRACE_JUDGES))} judge calls at most")
display(selection[["case_id", "intent", "difficulty", "final_answerability", "question"]])

RUN_IDS = []
if RUN_EVAL:
    # The full dataset is passed as the dataset object, which links the run to it in the Datasets tab
    data = eval_ds if subset_label == "full" else selection[["inputs", "expectations"]].reset_index(drop=True)
    for endpoint in ENDPOINTS:
        for rep in range(1, REPEATS + 1):
            name = f"{endpoint} · {subset_label.split(':')[0]} · {time.strftime('%Y-%m-%d %H:%M')}" + (f" · r{rep}" if REPEATS > 1 else "")
            with mlflow.start_run(run_name=name) as run:
                mlflow.set_tags({"endpoint": endpoint, "subset": subset_label, "dataset": DATASET_NAME,
                                 "judge_model": JUDGE_MODEL or "databricks-managed",
                                 "scorers_config_id": SCORERS_CONFIG_ID})
                mlflow.log_params({"n_cases": n, "repeat": rep, "excerpts_per_case": EXCERPTS_PER_CASE})
                if subset_label != "full":
                    mlflow.log_input(eval_ds, context="evaluation")   # links a sampled run to the golden dataset too
                mlflow.genai.evaluate(data=data, predict_fn=make_predict_fn(endpoint), scorers=SCORERS)
                RUN_IDS.append(run.info.run_id)
    print(f"✓ runs: {RUN_IDS}")
else:
    print("run_eval=false: nothing was evaluated.")

# COMMAND ----------

# DBTITLE 1,Unity Catalog tables — documented schemas; one run's rows are replaced on every write
from pyspark.sql.types import StructField, StructType

S, B, I, D, A = StringType(), BooleanType(), LongType(), DoubleType(), ArrayType(StringType())
METRICS = {
    "correctness": "expected facts present (built-in)",
    "fact_coverage": "expected facts present: full=1, partial=0.5, none=0",
    "fact_contradiction": "no contradiction with the expected facts",
    "refusal_handling": "correct refusals when the answer is not in the documentation",
    "relevance": "answer addresses the question",
    "language_match": "answer in the language of the question",
    "groundedness": "answer supported by the cited documents' excerpts (partial = 0.5)",
    "missed_answer": "no information missed that the cited excerpts contain",
    "retrieval_sufficiency": "excerpts contain what the expected answer needs",
    "document_recall": "expected documents returned or cited",
    "reference_integrity": "every cited document code exists",
    "expectations_guidelines": "case-specific guidelines followed",
    "call_ok": "endpoint answered",
    "latency_s": "response time in seconds (median in ka_eval_metrics)",
}
FAILURE_METRICS = ["correctness", "fact_contradiction", "refusal_handling", "relevance", "groundedness", "missed_answer",
                   "reference_integrity", "expectations_guidelines", "call_ok"]

RUNS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("run_name", S, "MLflow run name: endpoint · subset · date"),
    ("started_at", S, "Start time (UTC, ISO 8601) of the run"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("dataset_name", S, "Golden dataset (Unity Catalog MLflow dataset)"),
    ("subset", S, "full, sample:<n>:seed<seed> or cases:<ids>"),
    ("n_cases", I, "Cases evaluated"),
    ("repeat", I, "Repetition number (several repetitions measure the assistant's variability)"),
    ("judge_model", S, "Judge model used"),
    ("scorers_config_id", S, "Fingerprint of the scorer definitions and judge model: compare runs of the same configuration"),
    ("experiment_id", S, "MLflow experiment of the run"),
]
METRICS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("metric", S, "Scorer name"),
    ("meaning", S, "What the metric measures"),
    ("score", D, "Mean over the cases (1 = best); median response time for latency_s"),
    ("ci_low", D, "Lower bound of the 95% confidence interval"),
    ("ci_high", D, "Upper bound of the 95% confidence interval"),
    ("p95", D, "95th percentile (latency_s only)"),
    ("n", I, "Cases with a value for this metric"),
]
RESULTS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("case_id", S, "Golden case id (question id of the dataset builder)"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("question", S, "Question of the case (last user message)"),
    ("intent", S, "Question type assigned by the dataset builder"),
    ("difficulty", S, "easy, medium or hard (dataset builder)"),
    ("final_answerability", S, "full, partial, none or out_of_scope: whether the documentation answers the question"),
    ("expected", S, "Expected facts (one per line), or the expected answer for refusal cases"),
    ("expected_sources", A, "Documents the answer should rely on"),
    ("answer", S, "Assistant answer, without citation text fragments, truncated to 4000 characters"),
    ("returned_refs", A, "Documents returned by the assistant as citations"),
    ("cited_refs", A, "Document codes cited in the answer text"),
    ("unverified_refs", A, "Cited codes found neither in the index nor in the excerpts"),
    ("trace_id", S, "MLflow trace of the case"),
    *[(m, D, f"Numeric score: {d}") for m, d in METRICS.items()],
    ("human_fact_coverage", D, "Human rating of the expected facts present (full=1, partial=0.5, none=0)"),
    ("failed_scorers", A, "Scorers that failed on this case (score 0)"),
]
ASSESSMENTS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("case_id", S, "Golden case id"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("trace_id", S, "MLflow trace of the case"),
    ("assessment_name", S, "Scorer name, or the name of a human rating"),
    ("source_type", S, "LLM_JUDGE, CODE or HUMAN"),
    ("value", S, "Value as returned by the scorer"),
    ("value_numeric", D, "Numeric form: 1 = pass, 0 = fail, 0.5 = partial; counts and durations as is; NULL for labels"),
    ("rationale", S, "Rationale of the scorer"),
    ("error", S, "Error message when the scorer failed"),
]
TABLES = {
    RUNS_TABLE: (RUNS_COLUMNS, "Qualibot Knowledge Assistant evaluation runs on the golden dataset: one row per run."),
    METRICS_TABLE: (METRICS_COLUMNS, "Qualibot evaluation metrics: one row per run and metric, with 95% confidence intervals."),
    RESULTS_TABLE: (RESULTS_COLUMNS, "Qualibot evaluation results: one row per run and golden case, one column per metric."),
    ASSESSMENTS_TABLE: (ASSESSMENTS_COLUMNS, "Qualibot evaluation assessments: one row per run, golden case and scorer, "
                                             "human ratings included."),
}
SCHEMAS = {t: StructType([StructField(n, dt) for n, dt, _ in cols]) for t, (cols, _) in TABLES.items()}
VIEWS = {
    "v_ka_eval_metrics": ("Evaluation metrics with their run context, one row per run and metric", f"""
        SELECT r.started_at, r.run_name, r.endpoint, r.subset, r.n_cases, r.judge_model, r.scorers_config_id,
               m.metric, m.meaning, m.score, m.ci_low, m.ci_high, m.p95, m.n, r.run_id
        FROM {RUNS_TABLE} r JOIN {METRICS_TABLE} m USING (run_id)"""),
    "v_ka_eval_case_history": ("Per-case results across evaluation runs: spot regressions and unstable cases", f"""
        SELECT r.started_at, r.subset, r.scorers_config_id, c.*
        FROM {RUNS_TABLE} r JOIN {RESULTS_TABLE} c ON c.run_id = r.run_id"""),
    "v_quality_shared_scorers": ("Scorers shared by the evaluation and the production monitoring, side by side: "
                                 "daily means in production, run means in evaluation", f"""
        SELECT 'production' AS context, DATE(created_at) AS day, endpoint_name AS endpoint, assessment_name,
               COUNT(*) AS n, AVG(value_numeric) AS mean_value, CAST(NULL AS STRING) AS run_id
        FROM {PRODUCTION_ASSESSMENTS_TABLE}
        WHERE assessment_name IN ('relevance', 'language_match', 'groundedness', 'missed_answer', 'reference_integrity')
          AND error IS NULL
        GROUP BY ALL
        UNION ALL
        SELECT 'evaluation' AS context, DATE(r.started_at) AS day, a.endpoint, a.assessment_name,
               COUNT(*) AS n, AVG(a.value_numeric) AS mean_value, a.run_id
        FROM {ASSESSMENTS_TABLE} a JOIN {RUNS_TABLE} r USING (run_id)
        WHERE a.assessment_name IN ('relevance', 'language_match', 'groundedness', 'missed_answer', 'reference_integrity')
          AND a.error IS NULL AND a.source_type <> 'HUMAN'
        GROUP BY ALL"""),
}


def run_traces(run_id: str) -> list:
    """Traces of an evaluation run, with their tags, answer and assessments."""
    out = []
    for t in mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=run_id, return_type="list", max_results=2000):
        answer = getattr(getattr(t, "data", None), "response", None) or getattr(t.info, "response_preview", "") or ""
        if isinstance(answer, str) and answer.startswith('"'):
            try:
                answer = json.loads(answer)       # the trace output is stored JSON-encoded
            except json.JSONDecodeError:
                pass
        out.append({"trace_id": t.info.trace_id, "tags": t.info.tags or {}, "answer": answer,
                    "assessments": [assessment_row(a) for a in (t.info.assessments or [])
                                    if type(a).__name__ != "Expectation" and getattr(a, "expectation", None) is None]})
    return out


def collect(run_id: str) -> pd.DataFrame:
    """One row per case: numeric scores (human ratings prefixed human::), rationales (_why), answer, case metadata."""
    rows = []
    for tr in run_traces(run_id):
        row, why = {"trace_id": tr["trace_id"], "case_id": tr["tags"].get("case_id"), "answer": tr["answer"],
                    "_tags": tr["tags"], "_assessments": tr["assessments"]}, {}
        for a in tr["assessments"]:
            value = None if a["error"] else numeric_value(a["name"], a["value"])
            if value is None:
                continue
            key = f"human::{a['name']}" if a["source_type"] == "HUMAN" else a["name"]
            row[key] = value
            if a["rationale"]:
                why[key] = a["rationale"]
        row["_why"] = why
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.merge(CASES.drop(columns=["inputs"], errors="ignore"), on="case_id", how="left") if len(df) else df


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: score, 95% confidence interval (Wilson for pass/fail scores), number of cases."""
    out = []
    for m, meaning in METRICS.items():
        if m not in df.columns or df[m].dropna().empty:
            continue
        s = df[m].dropna()
        if m == "latency_s":
            out.append({"metric": m, "meaning": meaning, "score": s.median(), "ci_low": None, "ci_high": None,
                        "p95": s.quantile(.95), "n": len(s)})
            continue
        p = s.mean()
        if len(s) < 2:
            lo = hi = None
        elif set(s.unique()) <= {0.0, 1.0}:
            lo, hi = wilson(p, len(s))
        else:
            half = 1.96 * s.std(ddof=1) / math.sqrt(len(s))
            lo, hi = max(p - half, 0.0), min(p + half, 1.0)
        out.append({"metric": m, "meaning": meaning, "score": p, "ci_low": lo, "ci_high": hi, "p95": None, "n": len(s)})
    return pd.DataFrame(out)


def _expected_text(exp) -> str:
    exp = exp or {}
    return "\n".join(exp.get("expected_facts") or []) or str(exp.get("expected_response") or "")


def write_run_tables(run_id: str, res: pd.DataFrame, summary: pd.DataFrame):
    """Writes (or rewrites) one evaluation run in the four tables, then refreshes the views."""
    run = mlflow.get_run(run_id)
    tags, params = run.data.tags, run.data.params
    endpoint = tags.get("endpoint")
    run_row = {"run_id": run_id, "run_name": run.info.run_name, "endpoint": endpoint,
               "started_at": pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").isoformat(),
               "dataset_name": tags.get("dataset"), "subset": tags.get("subset"), "n_cases": params.get("n_cases"),
               "repeat": params.get("repeat"), "judge_model": tags.get("judge_model"),
               "scorers_config_id": tags.get("scorers_config_id"), "experiment_id": EXPERIMENT_ID}
    metric_rows = [{"run_id": run_id, **r} for r in summary.to_dict("records")]
    result_rows, assessment_rows = [], []
    for _, r in res.iterrows():
        exp = r.get("expectations") if isinstance(r.get("expectations"), dict) else {}
        t = r["_tags"]
        result_rows.append({
            "run_id": run_id, "case_id": r["case_id"], "endpoint": endpoint, "question": r.get("question"),
            "intent": r.get("intent"), "difficulty": r.get("difficulty"),
            "final_answerability": r.get("final_answerability"), "expected": _expected_text(exp),
            "expected_sources": [clean_ref(d["doc_uri"]) for d in exp.get("expected_retrieved_context", [])],
            "answer": str(r.get("answer") or "")[:4000],
            "returned_refs": json.loads(t.get("returned_refs", "[]")), "cited_refs": json.loads(t.get("cited_refs", "[]")),
            "unverified_refs": json.loads(t.get("unverified_refs", "[]")), "trace_id": r["trace_id"],
            **{m: r.get(m) for m in METRICS}, "human_fact_coverage": r.get("human::fact_coverage"),
            "failed_scorers": [m for m in FAILURE_METRICS if r.get(m) == 0]})
        assessment_rows += [{"run_id": run_id, "case_id": r["case_id"], "endpoint": endpoint, "trace_id": r["trace_id"],
                             "assessment_name": a["name"], "source_type": a["source_type"],
                             "value": a["value"] if isinstance(a["value"], str) or a["value"] is None else json.dumps(a["value"]),
                             "value_numeric": numeric_value(a["name"], a["value"]), "rationale": a["rationale"],
                             "error": a["error"]} for a in r["_assessments"]]
    for table, rows in [(RUNS_TABLE, [run_row]), (METRICS_TABLE, metric_rows), (RESULTS_TABLE, result_rows),
                        (ASSESSMENTS_TABLE, assessment_rows)]:
        cols, comment = TABLES[table]
        ensure_table(table, SCHEMAS[table], comment, {n: d for n, _, d in cols})
        replace_rows(table, SCHEMAS[table], rows, ["run_id"])
    for name, (comment, query) in VIEWS.items():
        if name == "v_quality_shared_scorers" and not spark.catalog.tableExists(PRODUCTION_ASSESSMENTS_TABLE):
            continue
        try:
            spark.sql(f"CREATE OR REPLACE VIEW {OUTPUT_SCHEMA}.{name} COMMENT '{_sql_text(comment)}' AS {query}")
        except Exception as e:                  # a view created by another identity cannot be replaced
            print(f"⚠️ view {name} not refreshed: {str(e)[:150]}")
    print(f"✓ run {run_id} written to {RUNS_TABLE}, {METRICS_TABLE}, {RESULTS_TABLE} ({len(result_rows)} cases), "
          f"{ASSESSMENTS_TABLE} ({len(assessment_rows)} rows)")

# COMMAND ----------

# DBTITLE 1,Report — scores with confidence intervals, breakdowns, failures; written to the run and to the tables
REPORT_RUN_ID = ""   # empty = most recent run


def last_run_id():
    r = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["start_time DESC"], max_results=1)
    return r.iloc[0]["run_id"] if len(r) else None


def pct(x):
    return "" if x is None or x != x else f"{x:.0%}"


run_id = REPORT_RUN_ID or (RUN_IDS[-1] if RUN_IDS else last_run_id())
if not run_id:
    print("No evaluation run yet: set run_eval=true.")
else:
    res = collect(run_id)
    run = mlflow.get_run(run_id)
    summary = summarize(res)
    print(f"Run {run.info.run_name} ({run_id}) · {len(res)} cases")
    display(summary.assign(score=lambda d: [f"{v:.1f} s" if m == "latency_s" else pct(v) for m, v in zip(d.metric, d.score)],
                           ci=lambda d: [f"{pct(lo)} – {pct(hi)}" if lo is not None and lo == lo else ""
                                         for lo, hi in zip(d.ci_low, d.ci_high)])
                   [["metric", "meaning", "score", "ci", "n"]])
    print("95% CI: range that most likely contains the true score; it is wide on few cases, which prevents over-reading.")

    cols = [c for c in ["correctness", "fact_coverage", "groundedness", "retrieval_sufficiency", "document_recall"]
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
        mlflow.log_metrics({f"report/{r.metric}": float(r.score) for r in summary.itertuples()})
        mlflow.log_table(summary, "report/summary.json")
        if len(failures):
            mlflow.log_table(failures.drop(columns=["_why", "_tags", "_assessments", "expectations"], errors="ignore")
                             .astype(str), "report/failures.json")
    print("✓ report logged to the run (Metrics: report/*, Artifacts: report/)")
    write_run_tables(run_id, res, summary)

# COMMAND ----------

# DBTITLE 1,Run comparison — from ka_eval_metrics; compare runs of the same subset and scorer configuration
if spark.catalog.tableExists(METRICS_TABLE):
    history = spark.sql(f"""
        SELECT r.started_at, r.run_name, r.subset, r.scorers_config_id, m.metric, m.score
        FROM {RUNS_TABLE} r JOIN {METRICS_TABLE} m USING (run_id)
        ORDER BY r.started_at DESC""").toPandas()
    view = (history.pivot_table(index=["started_at", "run_name", "subset", "scorers_config_id"], columns="metric",
                                values="score").sort_index(ascending=False).head(20).round(3))
    display(view.reset_index())
    print("Compare runs of the same subset and scorer configuration only. "
          "On 20-30 cases, differences below ~10 points are noise.")
else:
    print("No evaluation run written yet.")

# COMMAND ----------

# DBTITLE 1,Human labels — rate a few answers; agreement with the judges tells whether the scores can be trusted
# Each rating is stored in MLflow as a HUMAN assessment named "fact_coverage" on the trace, next to the judge's own
# "fact_coverage" verdict; "Save" also writes them to the result tables. The same labels feed the judge alignment below.
# Agreement ≥ 85% → the judge is reliable.
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
        save = widgets.Button(description="💾 Save ratings to Unity Catalog", button_style="success")
        save.on_click(lambda _: self._save())
        self.box = widgets.VBox([self.card, widgets.HBox(buttons + [skip, save]), self.stats])
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

    def _save(self):
        """Rewrites the run in the result tables, human ratings included (human_fact_coverage, HUMAN assessments)."""
        res = collect(label_run)
        write_run_tables(label_run, res, summarize(res))
        self.stats.value = agreement_report(res) + "<br>✓ saved to Unity Catalog"

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
    exp_by_case = dict(zip(CASES["case_id"], CASES["expectations"]))
    summary = summarize(res)
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
    out = [f"# Knowledge Assistant evaluation — {run.info.run_name}", f"- run_id: `{rid}`",
           f"- subset: {run.data.tags.get('subset', '?')} · {len(res)} cases · judge model: {run.data.tags.get('judge_model', '?')}",
           "", "## Summary", "", "| metric | meaning | score | 95% CI | n |", "|---|---|---|---|---|"]
    out += [f"| {m.metric} | {cell(m.meaning)} | {f'{m.score:.1f} s' if m.metric == 'latency_s' else pct(m.score)} | "
            f"{f'{pct(m.ci_low)} – {pct(m.ci_high)}' if m.ci_low is not None and m.ci_low == m.ci_low else ''} | {m.n} |"
            for m in summary.itertuples()]
    for _, r in res.sort_values("case_id").iterrows():
        exp = exp_by_case.get(r["case_id"], {}) or {}
        tags = r["_tags"]
        answer = str(r.get("answer") or "")
        out += ["", "---", "", f"### #{r['case_id']} · {r.get('intent')} · {r.get('difficulty')} · "
                f"answer in documentation: {r.get('final_answerability')}",
                f"- trace: `{r['trace_id']}` · latency: {tags.get('latency_s', '?')} s", "", f"**Question**: {r.get('question')}", ""]
        out += (["**Expected facts**:"] + [f"- {f}" for f in exp["expected_facts"]]) if exp.get("expected_facts") \
            else [f"**Expected answer**: {exp.get('expected_response', '')}"]
        if exp.get("guidelines"):
            out += ["", "**Guidelines**:"] + [f"- {g}" for g in exp["guidelines"]]
        out += ["", f"**Expected documents**: {', '.join(clean_ref(d['doc_uri']) for d in exp.get('expected_retrieved_context', [])) or '—'}",
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
