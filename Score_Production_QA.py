# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Production Answer Quality Monitoring
# MAGIC
# MAGIC Scores every new assistant turn of the Qualibot Knowledge Assistants with MLflow judges and scorers, twice a day.
# MAGIC The notebook never calls the assistants: it replays the answers already stored in `chat_messages`.
# MAGIC
# MAGIC ### Scorers
# MAGIC | Scorer | Type | Values | When |
# MAGIC |---|---|---|---|
# MAGIC | `question_intent`, `question_topic` | LLM judge | what the user asks (dashboard breakdowns) | every turn |
# MAGIC | `answer_type` | LLM judge | answered, partial_answer, not_found, out_of_scope_refusal, clarification_request, error_or_empty | every turn |
# MAGIC | `relevance` ¹ | LLM judge | yes / no — follow-up requests ("shorter", "remove document X") are judged against the previous turn | every turn |
# MAGIC | `completeness` | LLM judge | full / partial / none / not_applicable | every turn |
# MAGIC | `language_match` ¹ | LLM judge | yes / no | every turn |
# MAGIC | `safety` | built-in LLM judge | yes / no | every turn |
# MAGIC | `user_reaction` | LLM judge | implicit feedback carried by the user's next message | when the user wrote again (otherwise `no_next_turn`, free) |
# MAGIC | `groundedness` ¹ | LLM judge on the `RETRIEVER` step | supported / partially_supported / not_supported — the excerpts are a subset of the documents: absence of evidence is not contradiction | when the answer cites indexed documents |
# MAGIC | `missed_answer` ¹ | LLM judge on the `RETRIEVER` step | yes when the answer says "not found" (or leaves a part unanswered) while the excerpts contain it | when the answer cites indexed documents |
# MAGIC | `reference_integrity` ¹, `citation_count` | code | every cited code exists; number of documents listed as sources | every turn |
# MAGIC
# MAGIC ¹ identical in the evaluation notebook, so that production and golden-dataset results can be compared.
# MAGIC The rule-based **turn verdict** (`good` / `acceptable` / `bad`, with actionable `failure_reasons`) combines them.
# MAGIC Judges read the answers without the text fragments of citation links (`#:~:text=…`), which are longer than the
# MAGIC cited passages themselves.
# MAGIC
# MAGIC ### Outputs
# MAGIC | Where | Content |
# MAGIC |---|---|
# MAGIC | `chat_quality_scores` | one row per assistant turn: labels, verdicts, rationales, references, cost — replaced when a turn is re-scored |
# MAGIC | `chat_quality_assessments` | one row per turn × scorer (value, numeric value, rationale, error): any scorer can be charted without schema change |
# MAGIC | `chat_quality_scoring_runs` | one row per run: volumes, rates, estimated cost, judge/user agreement |
# MAGIC | `v_chat_quality_daily`, `v_chat_quality_scorers_daily`, `v_chat_quality_failure_reasons`, `v_chat_quality_review_queue` | dashboard-ready views |
# MAGIC | MLflow experiment | **Runs**: one per scoring run (rates, failure reasons, worst turns) · **Traces**: one per turn (conversation, answer, cited excerpts, every verdict, user vote), grouped by conversation in **Sessions** · **Judges / Scorers**: every scorer above, registered but not scheduled |
# MAGIC
# MAGIC Every table and column carries a Unity Catalog comment.
# MAGIC
# MAGIC ### Operations
# MAGIC - `dry_run=true` estimates the number of judge calls and the cost, and writes nothing.
# MAGIC - `test_limit` caps the number of turns of an ad-hoc run; `reset_outputs=true` drops the output tables first.
# MAGIC - `judge_config_id` fingerprints the scorers, the judge model and the verdict rules; `rescore_changed_config=true`
# MAGIC   re-scores the turns judged with another configuration.
# MAGIC - `feedback_to_agent_traces=true` also attaches the verdict to the assistant's own trace (needs CAN_EDIT on the
# MAGIC   assistants' experiments).
# MAGIC - Alerts compare each assistant's daily bad-answer rate with its 7-day baseline and can fail the job to trigger its notifications.
# MAGIC - A human review queue is built automatically: judge/user disagreements plus a stable random calibration sample.

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
dbutils.widgets.text("test_limit", "")                                            # e.g. "20"; empty = no cap
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"])                   # true = estimate only, nothing written
dbutils.widgets.dropdown("rescore_changed_config", "false", ["true", "false"])    # true = re-score turns judged with another configuration
dbutils.widgets.dropdown("reset_outputs", "false", ["true", "false"])             # true = drop the output tables before scoring
dbutils.widgets.dropdown("feedback_to_agent_traces", "false", ["true", "false"])  # true = also attach the verdict to the assistant's trace
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")                 # chat_messages / chat_feedbacks
dbutils.widgets.text("output_schema", "uat_proj.qualibot")                        # output tables and views
dbutils.widgets.text("experiment_path", "/Shared/qualibot-quality-scoring")
dbutils.widgets.text("judge_endpoint", "databricks-gpt-5-6-luna")                 # empty = Databricks-managed judge model

TEST_LIMIT = int(dbutils.widgets.get("test_limit")) if dbutils.widgets.get("test_limit").strip() else None
DRY_RUN = dbutils.widgets.get("dry_run") == "true"
RESCORE_CHANGED = dbutils.widgets.get("rescore_changed_config") == "true"
RESET_OUTPUTS = dbutils.widgets.get("reset_outputs") == "true"
FEEDBACK_TO_AGENT_TRACES = dbutils.widgets.get("feedback_to_agent_traces") == "true"
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint").strip()

SOURCE_TABLE = f"{SOURCE_SCHEMA}.chat_messages"
FEEDBACK_TABLE = f"{SOURCE_SCHEMA}.chat_feedbacks"            # optional: used if it exists
SCORES_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scores"
ASSESSMENTS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_assessments"
SCORING_RUNS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scoring_runs"

# ── Scope of a run ──
CHAT_HISTORY_LIMIT = 10                # mirrors CHAT_MAX_HISTORY / _trim_history() in server/routers/chat.py
MIN_TURN_AGE_MINUTES = 60              # wait a bit so the user's next message (implicit feedback) exists
SAMPLE_RATE = 1.0                      # deterministic sampling on message_id (1.0 = every turn)
MAX_TURNS_PER_RUN = 3000               # guard-rail for backlogs
MAX_PARALLEL_TURNS = 4                 # turns scored in parallel (each turn runs its scorers in parallel too)

# ── Cited document excerpts (RETRIEVER step) ──
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"   # index holding ALL chunks (the ALL assistant's index)
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None                # source table of the index; None = read from the index definition
EXCERPTS_PER_TURN = 10
EXCERPT_MAX_CHARS = 1500

# ── Cost estimate (pay-per-token, DBU per 1M tokens) ──
DBU_PER_M_INPUT = 2.857
DBU_PER_M_OUTPUT = 17.143
USD_PER_DBU = 0.07                     # adjust to your contract price for model serving
CHARS_PER_TOKEN = 3.8
OUTPUT_TOKENS_PER_JUDGE_CALL = 600     # rationale + hidden reasoning of a "thinking" judge model

# ── Human review queue & alerts ──
HUMAN_REVIEW_SAMPLE_RATE = 0.03        # random calibration sample, on top of judge/user disagreements
ALERT_MIN_TURNS = 15                   # a day needs at least this many turns to raise an alert
ALERT_BAD_RATE_DELTA = 0.10            # ... AND a bad-rate increase of at least 10 points vs the 7-day baseline
ALERT_Z = 2.5                          # ... AND statistically significant (binomial z-score)
FAIL_JOB_ON_ALERT = False              # True = the job fails on alert → Databricks job notification

TRACE_ID_CANDIDATES = ["trace_id", "mlflow_trace_id", "request_id", "databricks_request_id"]
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX", "BG", "RO", "PL", "TN"]
print(f"dry_run={DRY_RUN} · rescore_changed_config={RESCORE_CHANGED} · reset_outputs={RESET_OUTPUTS} · "
      f"test_limit={TEST_LIMIT} · feedback_to_agent_traces={FEEDBACK_TO_AGENT_TRACES}")

# COMMAND ----------

# DBTITLE 1,Connections — workspace, MLflow experiment, output reset
import os
import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(MAX_PARALLEL_TURNS)
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"   # the replay needs no preliminary test call

import mlflow

# An MLflow experiment is a workspace object: its parent folder must exist
_parent = EXPERIMENT_PATH.rsplit("/", 1)[0]
w.workspace.mkdirs(_parent if _parent.startswith("/Workspace") else f"/Workspace{_parent}")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id

# The output tables belong to the job identity: resetting them from the job avoids ownership issues
if RESET_OUTPUTS and not DRY_RUN:
    for table in (SCORES_TABLE, ASSESSMENTS_TABLE, SCORING_RUNS_TABLE):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    print(f"Output tables dropped: {SCORES_TABLE}, {ASSESSMENTS_TABLE}, {SCORING_RUNS_TABLE}")
print(f"MLflow {mlflow.__version__} · experiment {EXPERIMENT_PATH} (id {EXPERIMENT_ID})")

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
LABEL_SCORERS = {"question_intent", "question_topic", "answer_type", "user_reaction"}   # categorical: no numeric form
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

# DBTITLE 1,Document references — keys insensitive to language suffix, separators and zero padding
_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|txt)$", re.I)
_LANG = re.compile(r"[-_. ](%s)$" % "|".join(LANG_SUFFIXES), re.I)
_CODE = re.compile(r"^(?=.*\d)(?=(?:.*[A-Z]){2})[A-Z][A-Z0-9_.\-]{2,28}( (%s))?$" % "|".join(LANG_SUFFIXES))
_BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def _strip(s) -> str:
    return _LANG.sub("", _EXT.sub("", str(s).strip().split("/")[-1]))


def base_ref(s) -> str:
    """Document key: PRLAT538_FR, prlat-538 GB and PRLAT538.FR → PRLAT538; IN_APO_006 (typo present in some
    documents) and IN_APO_0006 → INAPO6. Language variants of a document therefore share one key."""
    groups = re.findall(r"[A-Za-z]+|\d+", _strip(s).upper())
    return "".join(str(int(g)) if g.isdigit() else g for g in groups)


def exact_ref(s) -> str:
    """Exact form of a code (separators, case and language suffix ignored, zero padding kept)."""
    return re.sub(r"[^A-Z0-9]", "", _strip(s).upper())


def code_like(text) -> set:
    """Document codes cited in an answer: **REF** in bold and ?ref= parameters of links."""
    text = str(text or "")
    return {c.strip() for c in _BOLD.findall(text) + _REF_IN_URL.findall(text) if _CODE.match(c.strip().upper())}


def source_refs(sources_json) -> list:
    """Documents listed by the assistant in sources_json: [{"rank", "title", "url", "n"}, ...]."""
    try:
        items = json.loads(sources_json) if sources_json else []
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for s in items if isinstance(items, list) else []:
        if isinstance(s, dict):
            m = _REF_IN_URL.search(s.get("url") or "")
            ref = s.get("title") or (m.group(1) if m else None)
            if ref:
                out.append(str(ref).strip())
    return list(dict.fromkeys(out))


REFS_BY_BASE, EXACT_REFS = {}, set()
try:
    _src = REF_SOURCE_TABLE or w.vector_search_indexes.get_index(VS_INDEX).delta_sync_index_spec.source_table
    for r in spark.table(_src).select("REF").distinct().collect():
        if r.REF:
            REFS_BY_BASE.setdefault(base_ref(r.REF), set()).add(r.REF)
            EXACT_REFS.add(exact_ref(r.REF))
    print(f"Document index: {len(REFS_BY_BASE)} documents ({_src})")
except Exception as e:
    print(f"⚠️ Document index unavailable — excerpts, groundedness and reference checks disabled: {str(e)[:200]}")


def resolve_refs(refs) -> list:
    """Cited codes → exact REF values of the index (all language variants)."""
    return sorted({real for r in refs if len(base_ref(r)) >= 4 for real in REFS_BY_BASE.get(base_ref(r), set())})


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

# DBTITLE 1,Production scorers — conversation labels, completeness, safety, user reaction, citation count
from mlflow.genai.scorers import Safety

INTENTS = ["definition_acronym", "document_lookup", "procedure_howto", "rule_requirement", "requirement_compliance",
           "comparison_multi_doc", "link_or_navigation", "person_or_org", "chitchat_or_meta", "out_of_scope"]
ANSWER_TYPES = ["answered", "partial_answer", "not_found", "out_of_scope_refusal", "clarification_request", "error_or_empty"]

PRODUCTION_JUDGES = {
    "question_intent": (
        "What the last user message asks for.", Literal[tuple(INTENTS)], CONTEXT + """
Classify what the last user message asks for:
definition_acronym; document_lookup (find a document or template); procedure_howto; rule_requirement (rule, threshold,
deadline, responsibility); requirement_compliance (whether the company complies with a customer or standard requirement,
and which internal documents demonstrate it); comparison_multi_doc; link_or_navigation; person_or_org (a person, a team,
an organisation); chitchat_or_meta; out_of_scope."""),
    "question_topic": (
        "Topic of the last user message, in 2 to 5 English words.", str, CONTEXT + """
Give the topic of the last user message in 2 to 5 English words (e.g. "operator qualification", "FAI after site change")."""),
    "answer_type": (
        "Kind of answer given by the assistant.", Literal[tuple(ANSWER_TYPES)], CONTEXT + """
Classify the answer: answered; partial_answer; not_found (says the documentation does not contain it);
out_of_scope_refusal (declines an off-topic request); clarification_request (asks the user to clarify instead of
answering); error_or_empty."""),
    "completeness": (
        "How completely the answer covers the question (full / partial / none / not_applicable).",
        Literal["full", "partial", "none", "not_applicable"], CONTEXT + """
How completely does the answer cover the question? full; partial (some parts left unanswered); none;
not_applicable (refusal, "not found" or clarification request)."""),
}


@scorer(name="user_reaction", description="Implicit feedback carried by the user's next message: no_next_turn, moves_on, "
                                          "follow_up, rephrase_same_question, correction_or_complaint.")
def user_reaction(inputs, outputs, trace):
    """Classifies the user's next message. Free (no judge call) when the user wrote nothing after the answer."""
    from typing import Literal

    from mlflow.entities import Feedback
    from mlflow.genai.judges import make_judge

    next_message = (inputs or {}).get("next_user_message")
    if not next_message:
        return Feedback(value="no_next_turn", rationale="The user wrote nothing after this answer.")
    question = next((m["content"] for m in reversed(inputs["messages"]) if m["role"] == "user"), "")
    judge = make_judge(
        name="user_reaction",
        instructions="""{{ inputs }} holds a user's question to an assistant on quality documentation and the message the
user wrote right after the assistant's answer, which is {{ outputs }}. Classify that next message:
moves_on (new unrelated question, or thanks); follow_up (natural continuation of the topic); rephrase_same_question
(asks the same thing again: the answer did not help); correction_or_complaint (says the answer is wrong, incomplete or
unhelpful). Write the rationale in English, in one sentence.""",
        feedback_value_type=Literal["moves_on", "follow_up", "rephrase_same_question", "correction_or_complaint"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"question": question, "next_user_message": next_message}, outputs=outputs)


@scorer(name="citation_count", description="Number of documents the assistant listed as sources.")
def citation_count(trace):
    """Number of documents the assistant listed as sources."""
    import json

    return len(json.loads((trace.info.tags or {}).get("source_refs", "[]")))


def build_scorers(model):
    """(LLM judges called for every turn, trace judges called when applicable, code scorers)."""
    kw = {"model": model} if model else {}
    every_turn = [make_judge(name=n, description=d, feedback_value_type=t, instructions=i, **kw)
                  for n, (d, t, i) in PRODUCTION_JUDGES.items()] + shared_llm_judges(model) + [Safety(**kw)]
    return every_turn, [user_reaction, groundedness, missed_answer], [reference_integrity, citation_count]

# COMMAND ----------

# DBTITLE 1,Turn verdict — actionable rules on top of the scorers
def turn_verdict(v: dict) -> tuple:
    """(verdict, failure_reasons) from the scorer values. bad = the user was badly served; acceptable = minor issue."""
    bad, warn = [], []
    at, intent = v.get("answer_type"), v.get("question_intent")
    if v.get("safety") == "no":
        bad.append("unsafe")
    if at == "error_or_empty":
        bad.append("empty_or_error")
    if v.get("relevance") == "no":
        bad.append("off_topic")
    if at == "out_of_scope_refusal" and intent not in ("out_of_scope", "chitchat_or_meta"):
        bad.append("wrongful_refusal")
    if v.get("missed_answer") == "yes":
        bad.append("missed_answer_in_sources")
    if v.get("groundedness") == "not_supported":
        bad.append("unsupported_claims")
    elif v.get("groundedness") == "partially_supported":
        warn.append("partially_supported_claims")
    if at in ("answered", "partial_answer") and v.get("completeness") == "none":
        bad.append("does_not_answer")
    elif v.get("completeness") == "partial" or at == "partial_answer":
        warn.append("incomplete")
    if v.get("reference_integrity") is False:
        warn.append("unverified_reference")
    if v.get("language_match") == "no":
        warn.append("language_mismatch")
    if at in ("answered", "partial_answer") and intent not in ("out_of_scope", "chitchat_or_meta") \
            and not v.get("citation_count"):
        warn.append("no_citation")
    if v.get("user_reaction") == "correction_or_complaint":
        warn.append("user_complaint")
    elif v.get("user_reaction") == "rephrase_same_question":
        warn.append("user_rephrased")
    return ("bad" if bad else "acceptable" if warn else "good"), bad + warn

# COMMAND ----------

# DBTITLE 1,Judge check and registration — the judge model answers, otherwise fall back to the Databricks-managed judge
import inspect

JUDGE_MODEL = f"databricks:/{JUDGE_ENDPOINT}" if JUDGE_ENDPOINT else None
_sample = {"inputs": {"messages": [{"role": "user", "content": "Quelle est la durée de conservation des enregistrements ?"}],
                      "next_user_message": None},
           "outputs": "Selon **QP-1457**, les enregistrements d'inspection sont conservés 10 ans."}


def _judge_works(model) -> bool:
    try:
        build_scorers(model)[0][0](**_sample)
        return True
    except Exception as e:
        print(f"⚠️ judge model {model or 'Databricks-managed'} failed: {str(e)[:300]}")
        return False


if JUDGE_MODEL and not DRY_RUN and not _judge_works(JUDGE_MODEL):
    print("   → falling back to the Databricks-managed judge model.")
    JUDGE_MODEL = None
LLM_JUDGES, TRACE_JUDGES, CODE_SCORERS = build_scorers(JUDGE_MODEL)
SCORERS = LLM_JUDGES + TRACE_JUDGES + CODE_SCORERS

# Fingerprint of everything that determines a score: scorers, judge model, verdict rules, excerpts, history window
JUDGE_CONFIG_ID = scorers_config_id(SCORERS, JUDGE_MODEL, inspect.getsource(turn_verdict),
                                    EXCERPTS_PER_TURN, CHAT_HISTORY_LIMIT)
print(f"Judge model: {JUDGE_MODEL or 'Databricks-managed'} · {len(SCORERS)} scorers · judge_config_id={JUDGE_CONFIG_ID}")
if not DRY_RUN:
    publish_scorers(SCORERS, EXPERIMENT_ID, JUDGE_CONFIG_ID)

# COMMAND ----------

# DBTITLE 1,Inputs — assistant turns to score, with thread, next user message and votes
from pyspark.sql import functions as F

src_cols = set(spark.table(SOURCE_TABLE).columns)
TRACE_COL = next((c for c in TRACE_ID_CANDIDATES if c in src_cols), None)
print(f"Trace id column: {TRACE_COL or 'none (feedback to the assistant traces disabled)'}")
if FEEDBACK_TO_AGENT_TRACES and not TRACE_COL:
    FEEDBACK_TO_AGENT_TRACES = False

df_pairs = spark.sql(f"""
    WITH msgs AS (
        SELECT
            id, created_at, session_id, division, role, content, sources_json, endpoint_name,
            {TRACE_COL if TRACE_COL else 'CAST(NULL AS STRING)'} AS trace_id,
            -- the app prepends a division instruction to user messages: strip it
            CASE WHEN role = 'user' AND content LIKE '[Division:%' AND LOCATE('state it explicitly.', content) > 0
                THEN REGEXP_REPLACE(SUBSTRING(content, LOCATE('state it explicitly.', content) + 20), '^[\\\\n\\\\r ]+', '')
                ELSE content
            END AS clean_content
        FROM {SOURCE_TABLE}
        WHERE status = 'ok' AND deleted = false
    ),
    threaded AS (
        SELECT *,
            COLLECT_LIST(STRUCT(role AS role, clean_content AS content)) OVER (
                PARTITION BY session_id ORDER BY created_at, id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_messages,
            LEAD(role) OVER (PARTITION BY session_id ORDER BY created_at, id) AS next_role,
            LEAD(clean_content) OVER (PARTITION BY session_id ORDER BY created_at, id) AS next_content
        FROM msgs
    )
    SELECT
        id AS message_id, created_at, session_id, division, endpoint_name, CAST(trace_id AS STRING) AS trace_id,
        content AS answer, sources_json, prior_messages,
        CASE WHEN next_role = 'user' THEN next_content END AS next_user_message
    FROM threaded
    -- endpoint_name IS NOT NULL excludes "duplicate shared conversation" copies (no endpoint/trace attribution)
    WHERE role = 'assistant' AND endpoint_name IS NOT NULL AND size(prior_messages) > 0
      AND created_at <= current_timestamp() - INTERVAL {int(MIN_TURN_AGE_MINUTES)} MINUTES
""")

# Turns already scored successfully are skipped; turns whose scoring failed (turn_verdict NULL) are retried.
# With rescore_changed_config, turns scored under another judge configuration are scored again.
if spark.catalog.tableExists(SCORES_TABLE):
    scored = spark.table(SCORES_TABLE).filter(F.col("turn_verdict").isNotNull())
    if RESCORE_CHANGED:
        scored = scored.filter(F.col("judge_config_id") == JUDGE_CONFIG_ID)
    df_pairs = df_pairs.join(scored.select("message_id"), on="message_id", how="left_anti")
else:
    print(f"{SCORES_TABLE} does not exist yet — scoring the backlog (capped at {MAX_TURNS_PER_RUN}).")

# User votes (👍/👎) — optional
if spark.catalog.tableExists(FEEDBACK_TABLE):
    fb = (spark.table(FEEDBACK_TABLE).groupBy("message_id")
          .agg(F.max((F.col("vote") == "down").cast("int")).alias("_down"),
               F.max((F.col("vote") == "up").cast("int")).alias("_up"),
               F.concat_ws(" || ", F.collect_list("comment")).alias("feedback_comment"))
          .withColumn("feedback_vote", F.when(F.col("_down") == 1, "down").when(F.col("_up") == 1, "up"))
          .drop("_down", "_up"))
    df_pairs = df_pairs.join(fb, on="message_id", how="left")
else:
    df_pairs = (df_pairs.withColumn("feedback_vote", F.lit(None).cast("string"))
                        .withColumn("feedback_comment", F.lit(None).cast("string")))

if SAMPLE_RATE < 1.0:
    df_pairs = df_pairs.filter((F.abs(F.hash(F.col("message_id").cast("string"))) % 1000) < int(SAMPLE_RATE * 1000))

cap = TEST_LIMIT or MAX_TURNS_PER_RUN
df_pairs = df_pairs.orderBy(F.col("created_at").desc()).limit(cap)
MESSAGE_ID_TYPE = df_pairs.schema["message_id"].dataType
CREATED_AT_TYPE = df_pairs.schema["created_at"].dataType
pdf_pairs = df_pairs.toPandas()
print(f"{len(pdf_pairs)} assistant turn(s) to score (cap {cap}).")

# COMMAND ----------

# DBTITLE 1,Replayed turns — evaluation records and the traced replay (answer + cited document excerpts)
from mlflow.entities import Document


def to_messages(prior) -> list:
    """COLLECT_LIST(STRUCT(...)) comes back as dicts or Rows depending on the Spark Connect path."""
    out = []
    for m in prior if prior is not None else []:
        if hasattr(m, "asDict"):
            m = m.asDict()
        out.append({"role": m["role"], "content": m["content"] or ""})
    return out


def trim_history(messages: list, limit: int = CHAT_HISTORY_LIMIT) -> list:
    """Mirrors server/routers/chat.py::_trim_history: keep the last `limit` messages, then drop leading turns until
    the window opens on a user message — the context the assistant actually saw."""
    if limit <= 0 or len(messages) <= limit:
        return messages
    trimmed = messages[-limit:]
    while len(trimmed) > 1 and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return trimmed


def compact(messages: list, older_chars: int = 1200, recent_chars: int = 4000) -> list:
    """Conversation as given to the judges: citation text fragments removed, older messages truncated; the two most
    recent ones are kept almost whole because follow-up requests can only be judged against them."""
    n = len(messages)
    return [{"role": m["role"], "content": clean_answer(m["content"])[:recent_chars if i >= n - 2 else older_chars]}
            for i, m in enumerate(messages)]


def last_user_question(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _text(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


TURNS = {}
for _, row in pdf_pairs.iterrows():
    thread = compact(trim_history(to_messages(row["prior_messages"])))
    raw_answer = str(row["answer"] or "")
    TURNS[str(row["message_id"])] = {
        "row": row, "thread": thread, "question": last_user_question(thread), "answer": clean_answer(raw_answer),
        "next_user_message": (_text(row.get("next_user_message")) or "")[:800] or None,
        "source_refs": source_refs(row["sources_json"]), "cited_refs": sorted(code_like(raw_answer)),
    }
RECORDS = [{"inputs": {"messages": t["thread"], "next_user_message": t["next_user_message"], "message_id": mid}}
           for mid, t in TURNS.items()]


@mlflow.trace(name="cited_document_excerpts", span_type="RETRIEVER")
def cited_document_excerpts(query: str, cited_documents: list) -> list:
    """Excerpts of the cited documents most related to the question and the answer (Vector Search, filtered on the
    documents). Shown as a RETRIEVER step so that the retrieval judges can use them and they are readable in the trace."""
    real = resolve_refs(cited_documents)
    if not real:
        return []
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX, columns=VS_COLUMNS, query_text=query[:2000], query_type="HYBRID",
        num_results=EXCERPTS_PER_TURN, filters_json=json.dumps({"REF": real}))
    cols = [c.name for c in res.manifest.columns]
    rows = [dict(zip(cols, r)) for r in ((res.result.data_array if res.result else None) or [])]
    return [Document(id=f"{r.get('REF')}#{i}", page_content=str(r.get("chunk_text") or "")[:EXCERPT_MAX_CHARS],
                     metadata={"doc_uri": r.get("REF"), "section": str(r.get("semantic_headers") or "")[:200]})
            for i, r in enumerate(rows)]


@mlflow.trace(name="qualibot_turn", span_type="AGENT")
def replay_turn(messages, next_user_message, message_id):
    """Returns the answer stored in chat_messages; the trace carries the turn's identifiers and cited documents."""
    t = TURNS[message_id]
    row = t["row"]
    tags = {"message_id": message_id, "endpoint": str(row["endpoint_name"]), "division": str(row["division"]),
            "agent_trace_id": _text(row.get("trace_id")) or "", "judge_model": JUDGE_MODEL or "",
            "judge_config_id": JUDGE_CONFIG_ID, "source_refs": json.dumps(t["source_refs"], ensure_ascii=False),
            "cited_refs": json.dumps(t["cited_refs"], ensure_ascii=False)}
    docs = []
    refs = list(dict.fromkeys(t["source_refs"] + t["cited_refs"]))
    if refs and REFS_BY_BASE:
        try:
            docs = cited_document_excerpts(f"{t['question']}\n{t['answer'][:800]}", refs)
        except Exception as e:
            tags["retrieval_error"] = str(e)[:300]
    classes = classify_refs(t["cited_refs"], "\n".join(d.page_content for d in docs))
    tags.update({"excerpt_refs": json.dumps(sorted({d.metadata["doc_uri"] for d in docs}), ensure_ascii=False),
                 **{f"{k}_refs": json.dumps(v, ensure_ascii=False) for k, v in classes.items()}})
    mlflow.update_current_trace(tags=tags, metadata={"mlflow.trace.session": str(row["session_id"])})
    return t["answer"]

# COMMAND ----------

# DBTITLE 1,Run — scoring with mlflow.genai.evaluate (one trace per turn)
from datetime import datetime, timezone

import pandas as pd

RUN_TS = datetime.now(timezone.utc).isoformat(timespec="seconds")
t_start = time.time()
MLFLOW_RUN_ID = None
results = {}


def estimated_usage(t: dict) -> tuple:
    """(judge calls, input tokens, output tokens) of one turn, estimated from the prompt sizes."""
    prompt_chars = len(json.dumps(t["thread"], ensure_ascii=False)) + len(t["answer"]) + 2500
    retrieval = 2 if (t["source_refs"] or t["cited_refs"]) and REFS_BY_BASE else 0
    reaction = 1 if t["next_user_message"] else 0
    calls = len(LLM_JUDGES) + retrieval + reaction
    tokens_in = (len(LLM_JUDGES) * prompt_chars + retrieval * (EXCERPTS_PER_TURN * EXCERPT_MAX_CHARS + len(t["answer"]))
                 + reaction * 3000) / CHARS_PER_TOKEN
    return calls, tokens_in, calls * OUTPUT_TOKENS_PER_JUDGE_CALL


def cost_usd(tokens_in, tokens_out) -> float:
    return (tokens_in * DBU_PER_M_INPUT + tokens_out * DBU_PER_M_OUTPUT) / 1e6 * USD_PER_DBU


if not TURNS:
    print("Nothing new to score.")
elif DRY_RUN:
    usage = [estimated_usage(t) for t in TURNS.values()]
    calls, t_in, t_out = (sum(u[i] for u in usage) for i in range(3))
    print(f"DRY RUN (nothing is scored or written) · {len(TURNS)} turns · ~{calls} judge calls "
          f"({calls / len(TURNS):.1f} per turn) · ~{t_in / 1e6:.2f}M in / ~{t_out / 1e6:.2f}M out tokens · "
          f"≈ ${cost_usd(t_in, t_out):.2f}")
else:
    with mlflow.start_run(run_name=f"quality-scoring {RUN_TS}") as run:
        MLFLOW_RUN_ID = run.info.run_id
        mlflow.set_tags({"judge_model": JUDGE_MODEL or "databricks-managed", "judge_config_id": JUDGE_CONFIG_ID,
                         "source_table": SOURCE_TABLE})
        mlflow.log_params({"n_turns": len(TURNS), "test_limit": TEST_LIMIT, "excerpts_per_turn": EXCERPTS_PER_TURN})
        mlflow.genai.evaluate(data=RECORDS, predict_fn=replay_turn, scorers=SCORERS)

    # Scores are read back from the traces (tags and assessments only)
    for tr in mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=MLFLOW_RUN_ID, return_type="list",
                                   include_spans=False, max_results=len(TURNS) + 100):
        tags = tr.info.tags or {}
        if tags.get("message_id") in TURNS:
            results[tags["message_id"]] = {"trace_id": tr.info.trace_id, "tags": tags,
                                           "assessments": [assessment_row(a) for a in (tr.info.assessments or [])]}
    print(f"{len(results)}/{len(TURNS)} turns scored in {time.time() - t_start:.0f} s · run {MLFLOW_RUN_ID}")

# COMMAND ----------

# DBTITLE 1,Records — one row per turn and one per turn × scorer; verdict and user vote attached to the trace
from mlflow.entities import AssessmentSource, AssessmentSourceType

RULE_SOURCE = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="turn_verdict_rules")
USER_SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="chat_user")
REQUIRED = {"answer_type", "relevance", "question_intent"}      # without them the verdict is not meaningful


def in_calibration_sample(message_id) -> bool:
    """Stable pseudo-random sample (same turns selected at every run)."""
    h = int(hashlib.md5(str(message_id).encode()).hexdigest()[:8], 16)
    return (h % 10000) < HUMAN_REVIEW_SAMPLE_RATE * 10000


def _yes(v):
    return None if v is None else v == "yes"


def build_record(mid: str) -> dict:
    t = TURNS[mid]
    res = results.get(mid, {"trace_id": None, "tags": {}, "assessments": []})
    row, tags = t["row"], res["tags"]
    ok_rows = [a for a in res["assessments"] if not a["error"] and a["value"] is not None]
    v = {a["name"]: a["value"] for a in ok_rows}
    why = {a["name"]: a["rationale"] for a in ok_rows}
    errors = [f"{a['name']}: {a['error'][:200]}" for a in res["assessments"] if a["error"]]
    if not res["trace_id"]:
        errors.append("turn not scored")
    if tags.get("retrieval_error"):
        errors.append(f"vector_search: {tags['retrieval_error']}")
    verdict, reasons = turn_verdict(v) if REQUIRED <= set(v) else (None, ["judge_failed"])
    vote = _text(row.get("feedback_vote"))
    disagreement = (verdict == "good" and vote == "down") or (verdict == "bad" and vote == "up")
    review = bool(verdict) and bool(disagreement or in_calibration_sample(mid))
    calls, t_in, t_out = estimated_usage(t)
    ground = v.get("groundedness")
    return {
        "message_id": row["message_id"], "created_at": row["created_at"], "session_id": row["session_id"],
        "division": row["division"], "endpoint_name": row["endpoint_name"], "trace_id": _text(row.get("trace_id")),
        "scoring_trace_id": res["trace_id"], "mlflow_run_id": MLFLOW_RUN_ID,
        "user_question": t["question"], "thread_turn_count": len(t["thread"]), "answer": t["answer"][:4000],
        "next_user_message": t["next_user_message"],
        "citation_count": len(t["source_refs"]), "source_refs": t["source_refs"], "cited_refs": t["cited_refs"],
        "excerpt_refs": json.loads(tags.get("excerpt_refs", "[]")),
        "approximate_refs": json.loads(tags.get("approximate_refs", "[]")),
        "unindexed_refs": json.loads(tags.get("unindexed_refs", "[]")),
        "unverified_refs": json.loads(tags.get("unverified_refs", "[]")),
        "feedback_vote": vote, "feedback_comment": _text(row.get("feedback_comment")),
        "question_intent": v.get("question_intent"), "question_topic": v.get("question_topic"),
        "in_scope": None if not v.get("question_intent") else
                    ("no" if v["question_intent"] in ("out_of_scope", "chitchat_or_meta") else "yes"),
        "is_follow_up": sum(m["role"] == "user" for m in t["thread"]) > 1,
        "answer_type": v.get("answer_type"),
        "relevance__value": _yes(v.get("relevance")), "relevance__rationale": why.get("relevance"),
        "completeness_level": v.get("completeness"),
        "completeness__value": None if v.get("completeness") is None else v["completeness"] in ("full", "not_applicable"),
        "completeness__rationale": why.get("completeness"),
        "language_match__value": _yes(v.get("language_match")), "language_match__rationale": why.get("language_match"),
        "safety__value": _yes(v.get("safety")), "safety__rationale": why.get("safety"),
        "grounding_source": "cited_documents" if ground else "none", "groundedness_level": ground,
        "groundedness__value": None if ground is None else ground == "supported",
        "groundedness__rationale": why.get("groundedness"),
        "missed_answer": _yes(v.get("missed_answer")), "missed_answer_detail": why.get("missed_answer"),
        "next_turn_signal": v.get("user_reaction"), "next_turn_rationale": why.get("user_reaction"),
        "turn_verdict": verdict, "failure_reasons": reasons,
        "needs_human_review": review,
        "review_reason": "judge_vs_user_disagreement" if disagreement and review else "calibration_sample" if review else None,
        "golden_candidate": verdict == "bad" or vote == "down",
        "judge_model": JUDGE_MODEL or "databricks-managed", "judge_config_id": JUDGE_CONFIG_ID,
        "judge_errors": errors, "n_judge_calls": calls,
        "estimated_cost_usd": round(cost_usd(t_in, t_out), 6), "scored_at": RUN_TS,
    }


def assessment_records(mid: str, record: dict) -> list:
    """One row per scorer of the turn, plus the rule-based verdict and the user's vote."""
    base = {k: record[k] for k in ("message_id", "created_at", "endpoint_name", "division", "scoring_trace_id",
                                   "mlflow_run_id", "judge_config_id", "scored_at")}
    items = list(results.get(mid, {}).get("assessments", []))
    if record["turn_verdict"]:
        items.append({"name": "turn_verdict", "value": record["turn_verdict"], "source_type": "CODE", "error": None,
                      "rationale": ", ".join(record["failure_reasons"]) or "no issue"})
    if record["feedback_vote"] in ("up", "down"):
        items.append({"name": "user_vote", "value": record["feedback_vote"], "source_type": "HUMAN", "error": None,
                      "rationale": record["feedback_comment"]})
    return [{**base, "assessment_name": a["name"], "source_type": a["source_type"],
             "value": a["value"] if isinstance(a["value"], str) or a["value"] is None else json.dumps(a["value"]),
             "value_numeric": numeric_value(a["name"], a["value"]), "rationale": a["rationale"], "error": a["error"]}
            for a in items]


records = [build_record(mid) for mid in TURNS] if MLFLOW_RUN_ID else []
assessment_rows = [a for mid, r in zip(TURNS, records) for a in assessment_records(mid, r)] if records else []
for r in records:
    if not r["scoring_trace_id"]:
        continue
    try:
        if r["turn_verdict"]:
            mlflow.log_feedback(trace_id=r["scoring_trace_id"], name="turn_verdict", value=r["turn_verdict"],
                                rationale=", ".join(r["failure_reasons"]) or "no issue", source=RULE_SOURCE)
        if r["feedback_vote"] in ("up", "down"):
            mlflow.log_feedback(trace_id=r["scoring_trace_id"], name="user_vote", value=r["feedback_vote"],
                                rationale=r["feedback_comment"] or None, source=USER_SOURCE)
    except Exception as e:
        r["judge_errors"].append(f"verdict logging: {str(e)[:200]}")
df_final = pd.DataFrame(records)
if len(df_final):
    print("Verdicts:", df_final["turn_verdict"].value_counts(dropna=False).to_dict())
    print("Answer types:", df_final["answer_type"].value_counts(dropna=False).to_dict())
    print(f"{int(df_final['n_judge_calls'].sum())} judge calls ({df_final['n_judge_calls'].mean():.1f} per turn) · "
          f"estimated cost ${df_final['estimated_cost_usd'].sum():.3f} · "
          f"{int((df_final['judge_errors'].map(len) > 0).sum())} turn(s) with scorer errors")

# COMMAND ----------

# DBTITLE 1,Unity Catalog tables — documented schemas, rows replaced by key, run ledger
from pyspark.sql.types import StructField, StructType

S, B, I, D, A = StringType(), BooleanType(), LongType(), DoubleType(), ArrayType(StringType())
_VERDICT_DOC = "good / acceptable / bad, from the verdict rules of the scoring notebook; NULL when the judges failed"

SCORES_COLUMNS = [
    ("message_id", MESSAGE_ID_TYPE, "Assistant message id (chat_messages.id); one row per scored assistant turn"),
    ("created_at", CREATED_AT_TYPE, "Time of the assistant answer"),
    ("session_id", S, "Conversation id"),
    ("division", S, "Division selected in the app: ALL, AS or IS"),
    ("endpoint_name", S, "Knowledge Assistant serving endpoint that answered"),
    ("trace_id", S, "MLflow trace id of the assistant's own answer"),
    ("scoring_trace_id", S, "MLflow trace id of the scoring (scoring experiment, Traces tab)"),
    ("mlflow_run_id", S, "MLflow run of the scoring run that produced the row"),
    ("user_question", S, "Last user message before the answer (division prefix removed)"),
    ("thread_turn_count", I, "Messages in the conversation window seen by the assistant"),
    ("answer", S, "Assistant answer, without citation text fragments, truncated to 4000 characters"),
    ("next_user_message", S, "Message the user wrote after the answer, if any"),
    ("citation_count", I, "Documents listed by the assistant as sources"),
    ("source_refs", A, "Document codes listed by the assistant as sources"),
    ("cited_refs", A, "Document codes cited in the answer text"),
    ("excerpt_refs", A, "Documents whose excerpts were checked by the retrieval judges"),
    ("approximate_refs", A, "Cited codes resolved despite a typo (e.g. IN_APO_006 → IN_APO_0006)"),
    ("unindexed_refs", A, "Cited codes absent from the index but mentioned in the excerpts (documents outside the corpus)"),
    ("unverified_refs", A, "Cited codes found neither in the index nor in the excerpts (possibly invented)"),
    ("feedback_vote", S, "User vote on the answer: up, down or NULL"),
    ("feedback_comment", S, "User comment attached to the vote"),
    ("question_intent", S, "Judge label: kind of question (definition_acronym, document_lookup, requirement_compliance…)"),
    ("question_topic", S, "Judge label: topic of the question in 2 to 5 English words"),
    ("in_scope", S, "yes unless the intent is out_of_scope or chitchat_or_meta"),
    ("is_follow_up", B, "The conversation had earlier user messages"),
    ("answer_type", S, "Judge label: answered, partial_answer, not_found, out_of_scope_refusal, clarification_request, error_or_empty"),
    ("relevance__value", B, "Judge: the answer addresses the question"),
    ("relevance__rationale", S, "Rationale of the relevance judge"),
    ("completeness_level", S, "Judge: full, partial, none or not_applicable"),
    ("completeness__value", B, "completeness_level is full or not_applicable"),
    ("completeness__rationale", S, "Rationale of the completeness judge"),
    ("language_match__value", B, "Judge: the answer is in the language of the question"),
    ("language_match__rationale", S, "Rationale of the language judge"),
    ("safety__value", B, "Judge: no harmful content"),
    ("safety__rationale", S, "Rationale of the safety judge"),
    ("grounding_source", S, "cited_documents when the answer could be checked against excerpts, none otherwise"),
    ("groundedness_level", S, "Judge: supported, partially_supported or not_supported; NULL when nothing could be checked"),
    ("groundedness__value", B, "groundedness_level is supported"),
    ("groundedness__rationale", S, "Rationale of the groundedness judge (names the unsupported claims)"),
    ("missed_answer", B, "Judge: the answer says 'not found' while the cited excerpts contain the information"),
    ("missed_answer_detail", S, "What was missed, according to the judge"),
    ("next_turn_signal", S, "Judge label of the user's next message: no_next_turn, moves_on, follow_up, rephrase_same_question, correction_or_complaint"),
    ("next_turn_rationale", S, "Rationale of the user reaction judge"),
    ("turn_verdict", S, _VERDICT_DOC),
    ("failure_reasons", A, "Reasons of a bad or acceptable verdict (unsupported_claims, missed_answer_in_sources, …)"),
    ("needs_human_review", B, "In the human review queue (judge/user disagreement or calibration sample)"),
    ("review_reason", S, "judge_vs_user_disagreement or calibration_sample"),
    ("golden_candidate", B, "Bad verdict or down vote: candidate case for the golden dataset"),
    ("judge_model", S, "Judge model used"),
    ("judge_config_id", S, "Fingerprint of the scorers, judge model and verdict rules; compare scores of the same configuration"),
    ("judge_errors", A, "Scorer errors of the turn"),
    ("n_judge_calls", I, "Judge model calls made for the turn"),
    ("estimated_cost_usd", D, "Estimated judge cost of the turn (USD)"),
    ("scored_at", S, "Start time (UTC) of the scoring run"),
]
SCORES_SCHEMA = StructType([StructField(n, t) for n, t, _ in SCORES_COLUMNS])
SCORES_DOCS = {n: d for n, _, d in SCORES_COLUMNS}

ASSESSMENTS_COLUMNS = [
    ("message_id", MESSAGE_ID_TYPE, "Assistant message id (chat_messages.id)"),
    ("created_at", CREATED_AT_TYPE, "Time of the assistant answer"),
    ("endpoint_name", S, "Knowledge Assistant serving endpoint that answered"),
    ("division", S, "Division selected in the app"),
    ("scoring_trace_id", S, "MLflow trace id of the scoring"),
    ("mlflow_run_id", S, "MLflow run of the scoring run"),
    ("judge_config_id", S, "Fingerprint of the scorers, judge model and verdict rules"),
    ("scored_at", S, "Start time (UTC) of the scoring run"),
    ("assessment_name", S, "Scorer name (relevance, groundedness, …), turn_verdict or user_vote"),
    ("source_type", S, "LLM_JUDGE, CODE or HUMAN"),
    ("value", S, "Value as returned by the scorer"),
    ("value_numeric", D, "Numeric form: 1 = pass, 0 = fail, 0.5 = partial; counts as is; NULL for labels"),
    ("rationale", S, "Rationale of the scorer"),
    ("error", S, "Error message when the scorer failed"),
]
ASSESSMENTS_SCHEMA = StructType([StructField(n, t) for n, t, _ in ASSESSMENTS_COLUMNS])
ASSESSMENTS_DOCS = {n: d for n, _, d in ASSESSMENTS_COLUMNS}

RUNS_COLUMNS = [
    ("run_ts", S, "Start time (UTC) of the scoring run"),
    ("mlflow_run_id", S, "MLflow run of the scoring run"),
    ("judge_model", S, "Judge model used"),
    ("judge_config_id", S, "Fingerprint of the scorers, judge model and verdict rules"),
    ("n_messages", I, "Turns scored"),
    ("n_judge_calls", I, "Judge model calls"),
    ("estimated_cost_usd", D, "Estimated judge cost (USD)"),
    ("duration_s", D, "Duration of the run (seconds)"),
    ("n_judge_errors", I, "Turns with at least one scorer error"),
    ("good_rate", D, "Share of good verdicts"),
    ("bad_rate", D, "Share of bad verdicts"),
    ("refusal_rate", D, "Share of not_found and out_of_scope_refusal answers"),
    ("groundedness_rate", D, "Share of supported answers among those checked against excerpts"),
    ("grounding_coverage", D, "Share of answers checked against excerpts"),
    ("judge_user_agreement", D, "Agreement between the verdict (bad / not bad) and the user votes"),
    ("n_voted", I, "Scored turns with a user vote"),
    ("n_needs_human_review", I, "Turns added to the human review queue"),
]
RUNS_SCHEMA = StructType([StructField(n, t) for n, t, _ in RUNS_COLUMNS])
RUNS_DOCS = {n: d for n, _, d in RUNS_COLUMNS}


def rate(series, value=True):
    s = series.dropna()
    return round(float((s == value).mean()), 4) if len(s) else None


run_row = None
if len(df_final):
    ensure_table(SCORES_TABLE, SCORES_SCHEMA, "Qualibot production turns scored by LLM judges: one row per assistant "
                 "turn (labels, verdicts, rationales, references). Written by the quality scoring job.", SCORES_DOCS)
    ensure_table(ASSESSMENTS_TABLE, ASSESSMENTS_SCHEMA, "Qualibot production scoring: one row per assistant turn and "
                 "scorer, including the rule-based turn_verdict and the user's vote.", ASSESSMENTS_DOCS)
    ensure_table(SCORING_RUNS_TABLE, RUNS_SCHEMA, "Qualibot production scoring runs: volumes, rates and estimated cost.",
                 RUNS_DOCS)
    replace_rows(SCORES_TABLE, SCORES_SCHEMA, records, ["message_id"])
    replace_rows(ASSESSMENTS_TABLE, ASSESSMENTS_SCHEMA, assessment_rows, ["message_id"])

    voted = df_final[df_final["feedback_vote"].isin(["up", "down"]) & df_final["turn_verdict"].notna()]
    agree = round(float(((voted["turn_verdict"] != "bad") == (voted["feedback_vote"] == "up")).mean()), 4) if len(voted) else None
    run_row = {
        "run_ts": RUN_TS, "mlflow_run_id": MLFLOW_RUN_ID, "judge_model": JUDGE_MODEL or "databricks-managed",
        "judge_config_id": JUDGE_CONFIG_ID, "n_messages": int(len(df_final)),
        "n_judge_calls": int(df_final["n_judge_calls"].sum()),
        "estimated_cost_usd": round(float(df_final["estimated_cost_usd"].sum()), 6),
        "duration_s": round(time.time() - t_start, 1),
        "n_judge_errors": int((df_final["judge_errors"].map(len) > 0).sum()),
        "good_rate": rate(df_final["turn_verdict"], "good"), "bad_rate": rate(df_final["turn_verdict"], "bad"),
        "refusal_rate": rate(df_final["answer_type"].isin(["not_found", "out_of_scope_refusal"]).where(df_final["answer_type"].notna())),
        "groundedness_rate": rate(df_final["groundedness__value"]),
        "grounding_coverage": rate(df_final["grounding_source"] == "cited_documents"),
        "judge_user_agreement": agree, "n_voted": int(len(voted)),
        "n_needs_human_review": int(df_final["needs_human_review"].sum()),
    }
    replace_rows(SCORING_RUNS_TABLE, RUNS_SCHEMA, [run_row], ["mlflow_run_id"])
    print(f"✓ {len(records)} rows → {SCORES_TABLE} · {len(assessment_rows)} rows → {ASSESSMENTS_TABLE} · "
          f"1 row → {SCORING_RUNS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Dashboard views — daily quality, scorers over time, failure reasons, review queue
VIEWS = {
    "v_chat_quality_daily": ("Daily quality per assistant, division and judge configuration", f"""
        SELECT DATE(created_at) AS day, endpoint_name, division, judge_config_id,
               COUNT(*) AS n_turns,
               COUNT_IF(turn_verdict = 'good') AS n_good,
               COUNT_IF(turn_verdict = 'acceptable') AS n_acceptable,
               COUNT_IF(turn_verdict = 'bad') AS n_bad,
               AVG(IF(turn_verdict = 'bad', 1.0, 0.0)) AS bad_rate,
               AVG(IF(turn_verdict = 'good', 1.0, 0.0)) AS good_rate,
               AVG(CAST(groundedness__value AS DOUBLE)) AS groundedness_rate,
               AVG(CAST(missed_answer AS DOUBLE)) AS missed_answer_rate,
               AVG(IF(answer_type IN ('not_found', 'out_of_scope_refusal'), 1.0, 0.0)) AS refusal_rate,
               AVG(IF(SIZE(unverified_refs) > 0, 1.0, 0.0)) AS unverified_reference_rate,
               COUNT_IF(feedback_vote = 'up') AS n_votes_up,
               COUNT_IF(feedback_vote = 'down') AS n_votes_down,
               SUM(estimated_cost_usd) AS estimated_cost_usd
        FROM {SCORES_TABLE}
        WHERE turn_verdict IS NOT NULL
        GROUP BY ALL"""),
    "v_chat_quality_scorers_daily": ("Daily mean of every scorer (numeric form: 1 = pass, 0 = fail) per assistant", f"""
        SELECT DATE(created_at) AS day, endpoint_name, division, judge_config_id, assessment_name, source_type,
               COUNT(*) AS n, AVG(value_numeric) AS mean_value, COUNT_IF(error IS NOT NULL) AS n_errors
        FROM {ASSESSMENTS_TABLE}
        GROUP BY ALL"""),
    "v_chat_quality_labels_daily": ("Daily counts of the categorical labels (intent, answer type, user reaction)", f"""
        SELECT DATE(created_at) AS day, endpoint_name, division, assessment_name, value AS label, COUNT(*) AS n
        FROM {ASSESSMENTS_TABLE}
        WHERE assessment_name IN ('question_intent', 'answer_type', 'user_reaction', 'turn_verdict') AND error IS NULL
        GROUP BY ALL"""),
    "v_chat_quality_failure_reasons": ("One row per failure reason of a scored turn", f"""
        SELECT DATE(created_at) AS day, endpoint_name, division, question_intent, turn_verdict, reason,
               message_id, user_question, scoring_trace_id
        FROM {SCORES_TABLE} LATERAL VIEW explode(failure_reasons) r AS reason
        WHERE turn_verdict IS NOT NULL"""),
    "v_chat_quality_review_queue": ("Turns to review by a human, and candidate cases for the golden dataset", f"""
        SELECT created_at, endpoint_name, division, review_reason, golden_candidate, turn_verdict, failure_reasons,
               question_intent, user_question, answer, feedback_vote, feedback_comment, groundedness__rationale,
               missed_answer_detail, scoring_trace_id, message_id
        FROM {SCORES_TABLE}
        WHERE needs_human_review OR golden_candidate"""),
}

if spark.catalog.tableExists(SCORES_TABLE) and spark.catalog.tableExists(ASSESSMENTS_TABLE):
    for name, (comment, query) in VIEWS.items():
        spark.sql(f"CREATE OR REPLACE VIEW {OUTPUT_SCHEMA}.{name} COMMENT '{_sql_text(comment)}' AS {query}")
    print(f"✓ views: {', '.join(f'{OUTPUT_SCHEMA}.{n}' for n in VIEWS)}")

# COMMAND ----------

# DBTITLE 1,Alerts — today's bad rate vs 7-day baseline, per assistant
alerts = []
if spark.catalog.tableExists(SCORES_TABLE):
    daily = spark.sql(f"""
        SELECT endpoint_name, DATE(created_at) AS day, COUNT(*) AS n,
               AVG(CASE WHEN turn_verdict = 'bad' THEN 1.0 ELSE 0.0 END) AS bad_rate
        FROM {SCORES_TABLE}
        WHERE judge_config_id = '{JUDGE_CONFIG_ID}' AND created_at >= current_date() - INTERVAL 8 DAYS
        GROUP BY endpoint_name, DATE(created_at)
    """).toPandas()
    for ep, g in daily.groupby("endpoint_name"):
        g = g.sort_values("day")
        last, base = g.iloc[-1], g.iloc[:-1]
        base = base[base["n"] >= 5]
        if last["n"] >= ALERT_MIN_TURNS and len(base) >= 3:
            baseline = float((base["bad_rate"] * base["n"]).sum() / base["n"].sum())
            p0 = min(max(baseline, 0.02), 0.98)
            z = (last["bad_rate"] - p0) / ((p0 * (1 - p0) / last["n"]) ** 0.5)
            if last["bad_rate"] - baseline > ALERT_BAD_RATE_DELTA and z > ALERT_Z:
                alerts.append(f"{ep}: bad rate {last['bad_rate']:.0%} on {last['day']} (n={int(last['n'])}) "
                              f"vs {baseline:.0%} over the previous days (z={z:.1f})")
print("\n".join("🚨 " + a for a in alerts) if alerts else "No alert.")

# COMMAND ----------

# DBTITLE 1,MLflow run — verdict metrics, failure reasons, worst turns; optional verdict on the assistant traces
if run_row:
    with mlflow.start_run(run_id=MLFLOW_RUN_ID):
        mlflow.set_tags({"alerts": " | ".join(alerts)[:4000] or "none"})
        mlflow.log_metrics({f"run/{k}": float(v) for k, v in run_row.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None})
        reasons = df_final.explode("failure_reasons")["failure_reasons"].value_counts()
        mlflow.log_metrics({f"reason/{k}": int(v) for k, v in reasons.items() if isinstance(k, str)})
        by_agent = df_final.groupby("endpoint_name")["turn_verdict"].apply(lambda s: float((s == "bad").mean()))
        mlflow.log_metrics({f"bad_rate/{k}": v for k, v in by_agent.items()})
        worst = df_final[df_final["turn_verdict"] == "bad"].head(50)
        if len(worst):
            mlflow.log_table(worst[["message_id", "endpoint_name", "question_intent", "user_question", "answer",
                                    "failure_reasons", "groundedness__rationale", "missed_answer_detail",
                                    "feedback_vote", "scoring_trace_id"]].astype(str), "worst_turns.json")
    print(f"✓ MLflow run {MLFLOW_RUN_ID}: metrics run/*, reason/*, bad_rate/*, artifact worst_turns.json")

if FEEDBACK_TO_AGENT_TRACES and len(df_final):
    src = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=f"qualibot-quality-scoring/{JUDGE_CONFIG_ID}")
    ok = ko = 0
    for r in df_final[df_final["trace_id"].notna() & df_final["turn_verdict"].notna()].to_dict("records"):
        try:
            mlflow.log_feedback(trace_id=r["trace_id"], name="qualibot_turn_verdict", value=r["turn_verdict"],
                                rationale=", ".join(r["failure_reasons"]) or "no issue", source=src)
            ok += 1
        except Exception:
            ko += 1   # typically: trace in an experiment without CAN_EDIT for the job identity
    print(f"Verdict on the assistant traces: {ok} attached · {ko} failed")

if FAIL_JOB_ON_ALERT and alerts:
    raise RuntimeError("Quality alert: " + " | ".join(alerts))

# COMMAND ----------

# DBTITLE 1,Run summary — what was written, and where to look
if DRY_RUN:
    print("Dry run: no judge call, no table write, no MLflow run. Set dry_run=false to score.")
elif df_final.empty:
    print("No new turn to score: tables and MLflow are unchanged.")
else:
    print(f"Tables : {SCORES_TABLE} ({spark.table(SCORES_TABLE).count()} turns in total), {ASSESSMENTS_TABLE}, "
          f"{SCORING_RUNS_TABLE}")
    print(f"Views  : {', '.join(f'{OUTPUT_SCHEMA}.{n}' for n in VIEWS)}")
    print(f"MLflow : {EXPERIMENT_PATH} → run {MLFLOW_RUN_ID} (Traces: one per turn; Judges: {len(SCORERS)} scorers)")
    print(f"Judge configuration: {JUDGE_CONFIG_ID}")
    display(spark.table(SCORES_TABLE).filter(F.col("scored_at") == RUN_TS)
            .select("created_at", "endpoint_name", "question_intent", "answer_type", "turn_verdict",
                    "failure_reasons", "groundedness_level", "user_question")
            .orderBy(F.col("created_at").desc()))
