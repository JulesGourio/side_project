# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Production Answer Quality Monitoring
# MAGIC
# MAGIC Scores every assistant turn answered by the Qualibot Knowledge Assistants with MLflow judges and scorers.
# MAGIC The notebook never calls the assistants: it replays the answers already stored in `chat_messages`.
# MAGIC
# MAGIC ### What appears in the MLflow experiment
# MAGIC | Tab | Content |
# MAGIC |---|---|
# MAGIC | **Runs** | one run per scoring run: verdict rates, failure reasons, bad rate per assistant, worst turns; the scored turns are the run's input dataset |
# MAGIC | **Traces** | one trace per scored turn — inputs = the conversation as the assistant saw it, output = the stored answer, a `RETRIEVER` step with the excerpts of the cited documents — with every verdict and its rationale, the rule-based `turn_verdict` and the user's vote; turns of a conversation are grouped in **Sessions** |
# MAGIC | **Judges / Scorers** | every scorer below, registered in the experiment (not scheduled: no background cost) |
# MAGIC
# MAGIC ### Scorers
# MAGIC | Scorer | Type | Values |
# MAGIC |---|---|---|
# MAGIC | `question_intent`, `question_topic` | LLM judge | what the user asks (drives the dashboard breakdowns) |
# MAGIC | `answer_type` | LLM judge | answered, partial_answer, not_found, out_of_scope_refusal, clarification_request, error_or_empty |
# MAGIC | `relevance` | LLM judge | yes / no — follow-up requests ("shorter", "remove document X") are judged against the previous turn |
# MAGIC | `completeness` | LLM judge | full / partial / none / not_applicable |
# MAGIC | `language_match` | LLM judge | yes / no |
# MAGIC | `user_reaction` | LLM judge | implicit feedback carried by the user's next message |
# MAGIC | `safety` | built-in LLM judge | yes / no |
# MAGIC | `groundedness` | LLM judge on the `RETRIEVER` step | supported / partially_supported / not_supported — excerpts are a subset of the documents: absence of evidence is not contradiction |
# MAGIC | `missed_answer` | LLM judge on the `RETRIEVER` step | yes when the answer says "not found" (or leaves a part unanswered) while the excerpts contain it |
# MAGIC | `reference_integrity`, `citation_count` | code | every cited code exists; number of documents listed as sources |
# MAGIC
# MAGIC The rule-based **turn verdict** (`good` / `acceptable` / `bad`, with actionable `failure_reasons`) combines them.
# MAGIC
# MAGIC ### Outputs (Unity Catalog)
# MAGIC - `chat_quality_scores`: one row per assistant turn, merged on `message_id`. Turns whose scoring failed are retried on the next run.
# MAGIC - `chat_quality_scoring_runs`: one row per run (volumes, rates, estimated cost, judge/user agreement).
# MAGIC
# MAGIC ### Operations
# MAGIC - `dry_run=true` estimates the number of judge calls and the cost, and writes nothing.
# MAGIC - `test_limit` caps the number of turns of an ad-hoc run; `reset_outputs=true` drops both output tables first.
# MAGIC - `judge_config_id` fingerprints the judges, the model and the rules; `rescore_changed_config=true` re-scores the turns
# MAGIC   judged with another configuration.
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
dbutils.widgets.dropdown("reset_outputs", "false", ["true", "false"])             # true = drop both output tables before scoring
dbutils.widgets.dropdown("feedback_to_agent_traces", "false", ["true", "false"])  # true = also attach the verdict to the assistant's trace
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")                 # chat_messages / chat_feedbacks
dbutils.widgets.text("output_schema", "uat_proj.qualibot")                        # chat_quality_scores / chat_quality_scoring_runs
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
SCORING_RUNS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scoring_runs"

# ── Scope of a run ──
CHAT_HISTORY_LIMIT = 10                # mirrors CHAT_MAX_HISTORY / _trim_history() in server/routers/chat.py
MIN_TURN_AGE_MINUTES = 60              # wait a bit so the user's next message (implicit feedback) exists
SAMPLE_RATE = 1.0                      # deterministic sampling on message_id (1.0 = every turn)
MAX_TURNS_PER_RUN = 3000               # guard-rail for backlogs
MERGE_BATCH_SIZE = 200                 # rows per MERGE into the scores table
MAX_PARALLEL_TURNS = 4                 # turns scored in parallel (each turn runs its scorers in parallel too)

# ── Cited document excerpts (RETRIEVER step) ──
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"   # index holding ALL chunks (the ALL assistant's index)
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None                # source table of the index; None = read from the index definition
EXCERPTS_PER_TURN = 12
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
import json
import os
import re
import time

import mlflow
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(MAX_PARALLEL_TURNS)
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"   # the replay needs no preliminary test call

# An MLflow experiment is a workspace object: its parent folder must exist
_parent = EXPERIMENT_PATH.rsplit("/", 1)[0]
w.workspace.mkdirs(_parent if _parent.startswith("/Workspace") else f"/Workspace{_parent}")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id

# The output tables belong to the job identity: resetting them from the job avoids ownership issues
if RESET_OUTPUTS and not DRY_RUN:
    for table in (SCORES_TABLE, SCORING_RUNS_TABLE):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    print(f"Output tables dropped: {SCORES_TABLE}, {SCORING_RUNS_TABLE}")
print(f"MLflow {mlflow.__version__} · experiment {EXPERIMENT_PATH} (id {EXPERIMENT_ID})")

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

# DBTITLE 1,Scorers — LLM judges (make_judge), built-in safety judge, retrieval judges and code scorers
from typing import Literal

from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Safety, scorer

INTENTS = ["definition_acronym", "document_lookup", "procedure_howto", "rule_requirement", "requirement_compliance",
           "comparison_multi_doc", "link_or_navigation", "person_or_org", "chitchat_or_meta", "out_of_scope"]
ANSWER_TYPES = ["answered", "partial_answer", "not_found", "out_of_scope_refusal", "clarification_request", "error_or_empty"]
NEXT_SIGNALS = ["no_next_turn", "moves_on", "follow_up", "rephrase_same_question", "correction_or_complaint"]

CONTEXT = """Qualibot is an internal assistant answering questions about the QUALITY documentation of an aerospace
manufacturer (procedures, work instructions, forms, templates, quality rules) for two divisions: AS (Aerostructures)
and IS (Interconnection Systems). Documents are identified by codes such as PRLAT508, QP-1457, NF-10065, INAQ619_FR.
It must answer only from those documents, cite them, answer in the user's language, and say so when the information is
not in the documentation.

{{ inputs }} holds `messages`, the conversation as the assistant saw it (oldest first; the last user message is the
question), and `next_user_message`, what the user wrote after the answer (null if nothing). {{ outputs }} is the
assistant answer under evaluation. Write the rationale in English, in one or two sentences.
"""

JUDGE_INSTRUCTIONS = {
    "question_intent": CONTEXT + """
Classify what the last user message asks for:
definition_acronym; document_lookup (find a document or template); procedure_howto; rule_requirement (rule, threshold,
deadline, responsibility); requirement_compliance (whether the company complies with a customer or standard requirement,
and which internal documents demonstrate it); comparison_multi_doc; link_or_navigation; person_or_org (a person, a team,
an organisation); chitchat_or_meta; out_of_scope.""",
    "question_topic": CONTEXT + """
Give the topic of the last user message in 2 to 5 English words (e.g. "operator qualification", "FAI after site change").""",
    "answer_type": CONTEXT + """
Classify the answer: answered; partial_answer; not_found (says the documentation does not contain it);
out_of_scope_refusal (declines an off-topic request); clarification_request (asks the user to clarify instead of
answering); error_or_empty.""",
    "relevance": CONTEXT + """
Does the answer address what the user asked, given the whole conversation? When the last user message asks to modify,
correct or restate the previous answer ("remove document X", "shorter", "same for Y"), judge whether the answer applies
that request. An appropriate refusal of an out-of-scope request, or a justified "not found", counts as relevant.
Return yes or no.""",
    "completeness": CONTEXT + """
How completely does the answer cover the question? full; partial (some parts left unanswered); none;
not_applicable (refusal, "not found" or clarification request).""",
    "language_match": CONTEXT + """
Is the answer written in the language of the last user message? Document titles and codes in another language do not
count. Return yes or no.""",
    "user_reaction": CONTEXT + """
Classify the user's next message: no_next_turn (null); moves_on (new unrelated question or thanks); follow_up (natural
continuation); rephrase_same_question (asks the same thing again: the answer did not help); correction_or_complaint
(says the answer is wrong, incomplete or unhelpful).""",
}
JUDGE_TYPES = {
    "question_intent": Literal[tuple(INTENTS)],
    "question_topic": str,
    "answer_type": Literal[tuple(ANSWER_TYPES)],
    "relevance": Literal["yes", "no"],
    "completeness": Literal["full", "partial", "none", "not_applicable"],
    "language_match": Literal["yes", "no"],
    "user_reaction": Literal[tuple(NEXT_SIGNALS)],
}


@scorer(name="groundedness")
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


@scorer(name="missed_answer")
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


@scorer(name="reference_integrity")
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


@scorer(name="citation_count")
def citation_count(trace):
    """Number of documents the assistant listed as sources."""
    import json

    return len(json.loads((trace.info.tags or {}).get("source_refs", "[]")))


def build_scorers(model):
    kw = {"model": model} if model else {}
    llm = [make_judge(name=n, instructions=JUDGE_INSTRUCTIONS[n], feedback_value_type=JUDGE_TYPES[n], **kw)
           for n in JUDGE_INSTRUCTIONS] + [Safety(**kw)]
    return llm, [groundedness, missed_answer], [reference_integrity, citation_count]

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
import hashlib
import inspect

from mlflow.genai.scorers import delete_scorer

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
LLM_JUDGES, RETRIEVAL_JUDGES, CODE_SCORERS = build_scorers(JUDGE_MODEL)
SCORERS = LLM_JUDGES + RETRIEVAL_JUDGES + CODE_SCORERS

# Fingerprint of everything that determines a score: any change produces a new identifier
JUDGE_CONFIG_ID = hashlib.sha1(json.dumps(
    [JUDGE_MODEL, JUDGE_INSTRUCTIONS, {k: str(v) for k, v in JUDGE_TYPES.items()},
     [s.model_dump().get("call_source") for s in RETRIEVAL_JUDGES + CODE_SCORERS],
     inspect.getsource(turn_verdict), EXCERPTS_PER_TURN, CHAT_HISTORY_LIMIT],
    sort_keys=True).encode()).hexdigest()[:12]


def publish_scorers(scorers, experiment_id):
    """Registers every scorer in the experiment (Judges / Scorers tab), replacing a registered scorer of the same name.
    Registration only: nothing is scheduled, so there is no background cost."""
    for s in scorers:
        try:
            try:
                delete_scorer(name=s.name, experiment_id=experiment_id)
            except Exception:
                pass                          # not registered yet
            s.register(name=s.name, experiment_id=experiment_id)
            print(f"  registered: {s.name}")
        except Exception as e:
            print(f"  not registered: {s.name} ({str(e)[:150]})")


print(f"Judge model: {JUDGE_MODEL or 'Databricks-managed'} · {len(SCORERS)} scorers · judge_config_id={JUDGE_CONFIG_ID}")
if not DRY_RUN:
    publish_scorers(SCORERS, EXPERIMENT_ID)

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
    """Older messages are truncated for the judges; the two most recent ones are kept almost whole because follow-up
    requests ("remove document X", "same for Y") can only be judged against them."""
    n = len(messages)
    return [{"role": m["role"], "content": m["content"][:recent_chars if i >= n - 2 else older_chars]}
            for i, m in enumerate(messages)]


def last_user_question(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _text(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


TURNS = {}
for _, row in pdf_pairs.iterrows():
    thread = compact(trim_history(to_messages(row["prior_messages"])))
    answer = str(row["answer"] or "")
    TURNS[str(row["message_id"])] = {
        "row": row, "thread": thread, "question": last_user_question(thread), "answer": answer,
        "next_user_message": (_text(row.get("next_user_message")) or "")[:800] or None,
        "source_refs": source_refs(row["sources_json"]), "cited_refs": sorted(code_like(answer)),
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

# DBTITLE 1,Run — scoring (mlflow.genai.evaluate), verdicts, votes
from datetime import datetime, timezone

import pandas as pd
from mlflow.entities import AssessmentSource, AssessmentSourceType

RUN_TS = datetime.now(timezone.utc).isoformat(timespec="seconds")
N_LLM_JUDGES = len(LLM_JUDGES)
t_start = time.time()
MLFLOW_RUN_ID = None
results = {}


def estimated_usage(t: dict) -> tuple:
    """(judge calls, input tokens, output tokens) of one turn, estimated from the prompt sizes."""
    prompt_chars = len(json.dumps(t["thread"], ensure_ascii=False)) + len(t["answer"]) + 2500
    calls = N_LLM_JUDGES + (len(RETRIEVAL_JUDGES) if (t["source_refs"] or t["cited_refs"]) and REFS_BY_BASE else 0)
    excerpt_chars = (calls - N_LLM_JUDGES) * EXCERPTS_PER_TURN * 1200
    tokens_in = (calls * prompt_chars + excerpt_chars) / CHARS_PER_TOKEN
    return calls, tokens_in, calls * OUTPUT_TOKENS_PER_JUDGE_CALL


def cost_usd(tokens_in, tokens_out) -> float:
    return (tokens_in * DBU_PER_M_INPUT + tokens_out * DBU_PER_M_OUTPUT) / 1e6 * USD_PER_DBU


def _value(a):
    v = getattr(a, "value", None)
    return getattr(v, "value", v)


def _error(a):
    fb = getattr(a, "feedback", None)
    err = getattr(fb, "error", None) if fb is not None else None
    return (getattr(err, "error_message", None) or str(err)) if err else None


if not TURNS:
    print("Nothing new to score.")
elif DRY_RUN:
    usage = [estimated_usage(t) for t in TURNS.values()]
    calls, t_in, t_out = (sum(u[i] for u in usage) for i in range(3))
    print(f"DRY RUN (nothing is scored or written) · {len(TURNS)} turns · ~{calls} judge calls · "
          f"~{t_in / 1e6:.2f}M in / ~{t_out / 1e6:.2f}M out tokens · ≈ ${cost_usd(t_in, t_out):.2f}")
else:
    with mlflow.start_run(run_name=f"quality-scoring {RUN_TS}") as run:
        MLFLOW_RUN_ID = run.info.run_id
        mlflow.set_tags({"judge_model": JUDGE_MODEL or "databricks-managed", "judge_config_id": JUDGE_CONFIG_ID,
                         "source_table": SOURCE_TABLE})
        mlflow.log_params({"n_turns": len(TURNS), "test_limit": TEST_LIMIT, "excerpts_per_turn": EXCERPTS_PER_TURN})
        mlflow.genai.evaluate(data=RECORDS, predict_fn=replay_turn, scorers=SCORERS)

    # One result per trace: scorer values, rationales and errors
    for tr in mlflow.search_traces(locations=[EXPERIMENT_ID], run_id=MLFLOW_RUN_ID, return_type="list", include_spans=False,
                                   max_results=len(TURNS) + 100):
        tags = tr.info.tags or {}
        if tags.get("message_id") not in TURNS:
            continue
        vals, why, errs = {}, {}, []
        for a in (tr.info.assessments or []):
            if _error(a):
                errs.append(f"{a.name}: {_error(a)[:200]}")
            elif _value(a) is not None:
                vals[a.name], why[a.name] = _value(a), getattr(a, "rationale", None)
        if tags.get("retrieval_error"):
            errs.append(f"vector_search: {tags['retrieval_error']}")
        results[tags["message_id"]] = {"trace_id": tr.info.trace_id, "tags": tags, "values": vals, "why": why, "errors": errs}
    print(f"{len(results)}/{len(TURNS)} turns scored in {time.time() - t_start:.0f} s · run {MLFLOW_RUN_ID}")

# COMMAND ----------

# DBTITLE 1,Records — one row per turn, verdict and user vote attached to the scoring trace
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
    t, res = TURNS[mid], results.get(mid, {"trace_id": None, "tags": {}, "values": {}, "why": {}, "errors": ["not scored"]})
    row, v, why, tags = t["row"], res["values"], res["why"], res["tags"]
    ok = REQUIRED <= set(v)
    verdict, reasons = turn_verdict(v) if ok else (None, ["judge_failed"])
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
        "judge_errors": res["errors"], "n_judge_calls": calls,
        "estimated_cost_usd": round(cost_usd(t_in, t_out), 6), "scored_at": RUN_TS,
    }


records = [build_record(mid) for mid in TURNS] if MLFLOW_RUN_ID else []
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
    print(f"Estimated cost ${df_final['estimated_cost_usd'].sum():.3f} · "
          f"{int((df_final['judge_errors'].map(len) > 0).sum())} turn(s) with scorer errors")

# COMMAND ----------

# DBTITLE 1,Output tables — explicit schema, MERGE on message_id, run ledger
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, LongType, StringType, StructField, StructType)

S, B, I, D, A = StringType(), BooleanType(), LongType(), DoubleType(), ArrayType(StringType())
OUT_SCHEMA = StructType([StructField(n, t) for n, t in [
    ("message_id", MESSAGE_ID_TYPE), ("created_at", CREATED_AT_TYPE), ("session_id", S), ("division", S),
    ("endpoint_name", S), ("trace_id", S), ("scoring_trace_id", S), ("mlflow_run_id", S),
    ("user_question", S), ("thread_turn_count", I), ("answer", S), ("next_user_message", S),
    ("citation_count", I), ("source_refs", A), ("cited_refs", A), ("excerpt_refs", A),
    ("approximate_refs", A), ("unindexed_refs", A), ("unverified_refs", A),
    ("feedback_vote", S), ("feedback_comment", S),
    ("question_intent", S), ("question_topic", S), ("in_scope", S), ("is_follow_up", B), ("answer_type", S),
    ("relevance__value", B), ("relevance__rationale", S),
    ("completeness_level", S), ("completeness__value", B), ("completeness__rationale", S),
    ("language_match__value", B), ("language_match__rationale", S),
    ("safety__value", B), ("safety__rationale", S),
    ("grounding_source", S), ("groundedness_level", S), ("groundedness__value", B), ("groundedness__rationale", S),
    ("missed_answer", B), ("missed_answer_detail", S),
    ("next_turn_signal", S), ("next_turn_rationale", S),
    ("turn_verdict", S), ("failure_reasons", A), ("needs_human_review", B), ("review_reason", S),
    ("golden_candidate", B), ("judge_model", S), ("judge_config_id", S), ("judge_errors", A),
    ("n_judge_calls", I), ("estimated_cost_usd", D), ("scored_at", S)]])


def _clean(v, field):
    """Python value → value accepted by the Spark field type (NaN/None handling, numpy scalars, casts)."""
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


def merge_rows(rows: list):
    sdf = spark.createDataFrame([tuple(_clean(r.get(f.name), f) for f in OUT_SCHEMA.fields) for r in rows], OUT_SCHEMA)
    if not spark.catalog.tableExists(SCORES_TABLE):
        sdf.write.saveAsTable(SCORES_TABLE)
        return
    sdf.createOrReplaceTempView("_scored_batch")
    spark.sql(f"""
        MERGE WITH SCHEMA EVOLUTION INTO {SCORES_TABLE} t
        USING _scored_batch s ON t.message_id = s.message_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def rate(series, value=True):
    s = series.dropna()
    return round(float((s == value).mean()), 4) if len(s) else None


run_row = None
if len(df_final):
    for start in range(0, len(records), MERGE_BATCH_SIZE):
        merge_rows(records[start:start + MERGE_BATCH_SIZE])
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
    RUNS_SCHEMA = StructType([StructField(k, t) for k, t in [
        ("run_ts", S), ("mlflow_run_id", S), ("judge_model", S), ("judge_config_id", S), ("n_messages", I),
        ("n_judge_calls", I), ("estimated_cost_usd", D), ("duration_s", D), ("n_judge_errors", I), ("good_rate", D),
        ("bad_rate", D), ("refusal_rate", D), ("groundedness_rate", D), ("grounding_coverage", D),
        ("judge_user_agreement", D), ("n_voted", I), ("n_needs_human_review", I)]])
    (spark.createDataFrame([tuple(_clean(run_row[f.name], f) for f in RUNS_SCHEMA.fields)], RUNS_SCHEMA)
          .write.mode("append").option("mergeSchema", "true").saveAsTable(SCORING_RUNS_TABLE))
    print(json.dumps(run_row, indent=1, default=str))

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
    print(f"Scored turns : {len(df_final)} → {SCORES_TABLE} ({spark.table(SCORES_TABLE).count()} rows in total)")
    print(f"Run ledger   : 1 row → {SCORING_RUNS_TABLE}")
    print(f"MLflow       : {EXPERIMENT_PATH} → run {MLFLOW_RUN_ID} (Traces tab: one trace per turn; Judges tab: {len(SCORERS)} scorers)")
    print(f"Judge config : {JUDGE_CONFIG_ID}")
    display(spark.table(SCORES_TABLE).filter(F.col("scored_at") == RUN_TS)
            .select("created_at", "endpoint_name", "question_intent", "answer_type", "turn_verdict",
                    "failure_reasons", "groundedness_level", "user_question")
            .orderBy(F.col("created_at").desc()))

# COMMAND ----------

# DBTITLE 1,Dashboard queries (Lakeview "ChatBot - Quality" page)
# MAGIC %md
# MAGIC ```sql
# MAGIC -- 1. Daily verdicts per assistant
# MAGIC SELECT DATE(created_at) AS day, endpoint_name, turn_verdict, COUNT(*) AS n
# MAGIC FROM uat_proj.qualibot.chat_quality_scores
# MAGIC GROUP BY ALL;
# MAGIC
# MAGIC -- 2. Root causes (what to fix first)
# MAGIC SELECT reason, COUNT(*) AS n
# MAGIC FROM uat_proj.qualibot.chat_quality_scores LATERAL VIEW explode(failure_reasons) r AS reason
# MAGIC WHERE created_at >= current_date() - INTERVAL 30 DAYS
# MAGIC GROUP BY reason ORDER BY n DESC;
# MAGIC
# MAGIC -- 3. Quality per type of question
# MAGIC SELECT question_intent, COUNT(*) AS n,
# MAGIC        AVG(CASE WHEN turn_verdict = 'bad' THEN 1 ELSE 0 END) AS bad_rate,
# MAGIC        AVG(CASE WHEN groundedness__value THEN 1 ELSE 0 END) AS groundedness,
# MAGIC        AVG(CASE WHEN answer_type IN ('not_found','out_of_scope_refusal') THEN 1 ELSE 0 END) AS refusal_rate
# MAGIC FROM uat_proj.qualibot.chat_quality_scores GROUP BY question_intent;
# MAGIC
# MAGIC -- 4. Documentation gaps: in-scope questions answered "not found" (topics to document or to index better)
# MAGIC SELECT question_topic, COUNT(*) AS n, max(user_question) AS example
# MAGIC FROM uat_proj.qualibot.chat_quality_scores
# MAGIC WHERE answer_type = 'not_found' AND in_scope = 'yes' GROUP BY question_topic ORDER BY n DESC;
# MAGIC
# MAGIC -- 5. Judge vs users (calibration)
# MAGIC SELECT feedback_vote, turn_verdict, COUNT(*) AS n
# MAGIC FROM uat_proj.qualibot.chat_quality_scores WHERE feedback_vote IS NOT NULL GROUP BY ALL;
# MAGIC
# MAGIC -- 6. Human review queue and golden-dataset candidates
# MAGIC SELECT created_at, endpoint_name, review_reason, user_question, answer, failure_reasons, feedback_vote, feedback_comment
# MAGIC FROM uat_proj.qualibot.chat_quality_scores
# MAGIC WHERE needs_human_review OR golden_candidate ORDER BY created_at DESC;
# MAGIC
# MAGIC -- 7. Estimated cost per day
# MAGIC SELECT DATE(run_ts) AS day, SUM(estimated_cost_usd) AS usd, SUM(n_messages) AS turns, SUM(n_judge_calls) AS judge_calls
# MAGIC FROM uat_proj.qualibot.chat_quality_scoring_runs GROUP BY ALL ORDER BY day;
# MAGIC ```
