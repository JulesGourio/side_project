# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Production Answer Quality Monitoring
# MAGIC
# MAGIC Scores the assistant turns stored in `chat_messages` with MLflow judges, twice a day. The assistants are not called.
# MAGIC
# MAGIC ### What the judges read
# MAGIC - **Conversation**: the messages the assistant saw (last 10), whole.
# MAGIC - **Answer**: the stored answer, without the text fragments of citation links (`#:~:text=…`).
# MAGIC - **Evidence** (`RETRIEVER` step): the passages the assistant retrieved from the `qualibot` index, read from its own
# MAGIC   MLflow trace (`assistant_retrieval`). When a trace shows no retrieval step, excerpts of every document the answer
# MAGIC   relies on, searched with each line that cites it (`cited_document_excerpts`). No limit, no truncation.
# MAGIC - **Independent search** (`corpus_search`): the question searched in the whole index, to tell a retrieval miss from a
# MAGIC   documentation gap.
# MAGIC
# MAGIC ### Scorers
# MAGIC | Scorer | Type | Values | When |
# MAGIC |---|---|---|---|
# MAGIC | `question_intent` | LLM judge | kind of question | every turn |
# MAGIC | `answer_type` | LLM judge | answered_full, answered_partial, not_found, out_of_scope_refusal, clarification_request, error_or_empty | every turn |
# MAGIC | `relevance` ¹ | LLM judge | yes / no | every turn |
# MAGIC | `language_match` ¹ | LLM judge | yes / no | every turn |
# MAGIC | `safety` | built-in LLM judge | yes / no | stable 10% sample |
# MAGIC | `user_reaction` | LLM judge | reaction carried by the user's next message | when the user wrote again |
# MAGIC | `groundedness` ¹ | LLM judge on the evidence | supported / partially_supported / not_supported | when there is evidence |
# MAGIC | `missed_answer` ¹ | LLM judge on the evidence | yes when the evidence holds what the answer says it did not find | when there is evidence |
# MAGIC | `retrieval_quality` ¹ | 1-2 LLM judges | sufficient / retrieval_miss / documentation_gap / not_applicable | when the assistant's retrieval is known |
# MAGIC | `compliance_claim` ¹ | LLM judge on the evidence | evidence_based / unsupported_compliance_claim / not_applicable | every turn |
# MAGIC | `reference_integrity` ¹, `citation_count` | code | every cited code exists; number of sources | every turn |
# MAGIC
# MAGIC ¹ identical in the evaluation notebook. The rule-based **turn verdict** (`good` / `acceptable` / `bad`) combines the
# MAGIC scorers into `failure_reasons` and the stage at fault, `error_source`: **retrieval** (the index holds what the
# MAGIC assistant did not retrieve) or **generation** (the model misused or went beyond what it retrieved).
# MAGIC
# MAGIC ### Outputs
# MAGIC | Where | Content |
# MAGIC |---|---|
# MAGIC | `chat_quality_scores` | one row per turn: labels, verdict, failure reasons, stage at fault, evidence, rationales, tokens, cost |
# MAGIC | `chat_quality_assessments` | one row per turn × scorer: value, numeric value, rationale, error |
# MAGIC | `chat_quality_scoring_runs` | one row per run: volumes, rates, cost, turns left |
# MAGIC | MLflow experiment | Runs (one per scoring run), Traces (one per turn, grouped by conversation), Judges (every scorer, not scheduled) |
# MAGIC
# MAGIC ### Operations
# MAGIC - Judge calls are paced to use `JUDGE_RATE_SHARE` of the judge model's token limits, measured with the tokens the
# MAGIC   model reports. Turns are scored in batches of about one minute of that budget and written after each batch; no
# MAGIC   batch starts after `max_run_minutes`, and the turns left are scored by the next run.
# MAGIC - The run stops if the judge model (`judge_endpoint`) does not answer.
# MAGIC - `dry_run=true`: estimated calls, tokens, cost and duration; nothing is written.
# MAGIC - `test_limit`: number of turns of an ad-hoc run. `reset_outputs=true`: scores again from scratch and replaces the
# MAGIC   tables at the first write.
# MAGIC - `rescore_changed_config=true`: scores again the turns judged with another `judge_config_id` (fingerprint of the
# MAGIC   scorers, judge model, evidence selection and verdict rules).
# MAGIC - `fail_on_alert=true`: a run whose daily bad-answer rate rises above the 7-day baseline fails after writing, which
# MAGIC   sends the job's failure e-mail.

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
dbutils.widgets.dropdown("reset_outputs", "false", ["true", "false"])             # true = score from scratch and replace the output tables
dbutils.widgets.dropdown("feedback_to_agent_traces", "false", ["true", "false"])  # true = also attach the verdict to the assistant's trace
dbutils.widgets.dropdown("fail_on_alert", "false", ["true", "false"])             # true = the run fails on a quality alert (job e-mail)
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")                 # chat_messages / chat_feedbacks
dbutils.widgets.text("output_schema", "uat_proj.qualibot")                        # output tables
dbutils.widgets.text("experiment_path", "/Shared/qualibot-quality-scoring")
dbutils.widgets.text("judge_endpoint", "databricks-gpt-6-luna")                   # judge model serving endpoint
dbutils.widgets.text("max_run_minutes", "100")      # time budget: no new batch of turns starts after it (backlog absorbed over runs)

TEST_LIMIT = int(dbutils.widgets.get("test_limit")) if dbutils.widgets.get("test_limit").strip() else None
DRY_RUN = dbutils.widgets.get("dry_run") == "true"
RESCORE_CHANGED = dbutils.widgets.get("rescore_changed_config") == "true"
RESET_OUTPUTS = dbutils.widgets.get("reset_outputs") == "true"
FEEDBACK_TO_AGENT_TRACES = dbutils.widgets.get("feedback_to_agent_traces") == "true"
FAIL_ON_ALERT = dbutils.widgets.get("fail_on_alert") == "true"
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint").strip()
MAX_RUN_MINUTES = float(dbutils.widgets.get("max_run_minutes") or 100)

SOURCE_TABLE = f"{SOURCE_SCHEMA}.chat_messages"
FEEDBACK_TABLE = f"{SOURCE_SCHEMA}.chat_feedbacks"            # optional: used if it exists
SCORES_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scores"
ASSESSMENTS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_assessments"
SCORING_RUNS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scoring_runs"

# ── Scope of a run ──
CHAT_HISTORY_LIMIT = 10                # mirrors CHAT_MAX_HISTORY / _trim_history() in server/routers/chat.py
MIN_TURN_AGE_MINUTES = 60              # wait a bit so the user's next message (implicit feedback) exists
SAMPLE_RATE = 1.0                      # deterministic sampling on message_id (1.0 = every turn)
MAX_TURNS_PER_RUN = 3000               # turns loaded per run; the time budget (max_run_minutes) usually stops earlier
MAX_PARALLEL_TURNS = 4                 # turns scored in parallel (each turn runs its scorers in parallel too)

# ── Evidence (RETRIEVER step): the passages the assistant retrieved, read from its own trace ──
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"   # index queried by the assistants
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None                # source table of the index; None = read from the index definition
EXCERPTS_PER_QUERY = 3                 # when the trace has no retrieval step: chunks per search of a cited document
CORPUS_SEARCH_RESULTS = 10             # chunks of the independent search of the whole index (retrieval_quality)

# ── Judge model rate limits (pay-per-token endpoint, shared with every other use of the model) ──
JUDGE_INPUT_TOKENS_PER_MINUTE = 200_000
JUDGE_OUTPUT_TOKENS_PER_MINUTE = 20_000
JUDGE_RATE_SHARE = 0.7                 # share of the limits used by this job; the rest stays available to other uses
JUDGE_MAX_RETRIES = 7                  # retries of a call rejected for rate limit (1 s, 2 s … 60 s: about 2 minutes)
# The request limits (1,000 per second, 360,000 per hour) are far above the calls of this job. The pacing uses the
# tokens reported by the judge model for each call.

# ── Cost (pay-per-token, DBU per 1M tokens), from the tokens reported by the judge model ──
DBU_PER_M_INPUT = 1.4
DBU_PER_M_OUTPUT = 7.1
USD_PER_DBU = 0.07                     # adjust to your contract price for model serving
# Estimates used before any call is measured (dry run, first batch)
CHARS_PER_TOKEN = 3.8
OUTPUT_TOKENS_PER_JUDGE_CALL = 400     # rationale + hidden reasoning
JUDGE_INSTRUCTIONS_CHARS = 2500        # instructions and context of a judge prompt
DEFAULT_CHUNK_CHARS = 2500             # mean chunk size, used when it cannot be read from the chunks table

# ── Human review queue & alerts ──
HUMAN_REVIEW_SAMPLE_RATE = 0.03        # random calibration sample, on top of judge/user disagreements
ALERT_MIN_TURNS = 15                   # a day needs at least this many turns to raise an alert
ALERT_BAD_RATE_DELTA = 0.10            # ... AND a bad-rate increase of at least 10 points vs the 7-day baseline
ALERT_Z = 2.5                          # ... AND statistically significant (binomial z-score)

TRACE_ID_CANDIDATES = ["trace_id", "mlflow_trace_id", "request_id", "databricks_request_id"]
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX", "BG", "RO", "PL", "TN"]
print(f"dry_run={DRY_RUN} · rescore_changed_config={RESCORE_CHANGED} · reset_outputs={RESET_OUTPUTS} · "
      f"test_limit={TEST_LIMIT} · feedback_to_agent_traces={FEEDBACK_TO_AGENT_TRACES} · fail_on_alert={FAIL_ON_ALERT}")

# COMMAND ----------

# DBTITLE 1,Connections — workspace, MLflow experiment
import os
import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(MAX_PARALLEL_TURNS)
os.environ["MLFLOW_GENAI_EVAL_MAX_RETRIES"] = str(JUDGE_MAX_RETRIES)      # rate-limited judge calls are retried
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"   # the replay needs no preliminary test call

import mlflow

# An MLflow experiment is a workspace object: its parent folder must exist
_parent = EXPERIMENT_PATH.rsplit("/", 1)[0]
w.workspace.mkdirs(_parent if _parent.startswith("/Workspace") else f"/Workspace{_parent}")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id
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
from mlflow.entities import Document
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
question); any other field of the inputs is an identifier to ignore. {{ outputs }} is the assistant answer under
evaluation. Write the rationale in English, in one or two sentences.
"""

SHARED_JUDGES = {
    "relevance": (
        "The answer addresses what the user asked, given the whole conversation.",
        CONTEXT + """
Does the answer address what the user asked, given the whole conversation? When the last user message asks to modify,
correct or restate the previous answer ("remove document X", "shorter", "same for Y"), judge whether the answer applies
that request. An appropriate refusal of an out-of-scope request, or a justified "not found", counts as relevant.
Judge only whether the content addresses the request: the language of the answer, the correctness of its facts and the
strength of its evidence are judged separately and must not lower this verdict. Return yes or no."""),
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


@scorer(name="groundedness", description="The answer's claims are supported by its evidence: the passages the "
                                         "assistant retrieved, or excerpts of the documents it cites "
                                         "(supported / partially_supported / not_supported).")
def groundedness(inputs, outputs, trace):
    """Checks the answer's factual claims against its evidence (RETRIEVER step): the passages the assistant retrieved
    when its trace exposes them, otherwise excerpts of the documents it cites. No assessment without evidence."""
    from typing import Literal

    from mlflow.entities import SpanType
    from mlflow.genai.judges import make_judge

    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    docs = [d if isinstance(d, dict) else d.to_dict()
            for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    if not docs:
        return None
    evidence = "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri')}] {d.get('page_content')}" for d in docs)
    if context.get("evidence_source") == "assistant_retrieval":
        scope = """The evidence is EVERY PASSAGE THE ASSISTANT RETRIEVED before answering: a claim these passages do not
state was not taken from the documentation (unless it is general knowledge or restates the question).
Return:
- supported: every claim is stated by the passages;
- partially_supported: a claim is only partly supported or close but not exact, or a secondary claim is absent;
- not_supported: a key claim (value, rule, compliance statement, document identity) is contradicted by the passages or
  absent from them."""
    else:
        scope = """The evidence is EXCERPTS of the documents the answer cites, only a SUBSET of those documents: a claim
absent from the excerpts is not verifiable, which is NOT a contradiction and does NOT lower the verdict.
Return:
- supported: every claim that the excerpts cover is stated by them, even if other claims cannot be checked;
- partially_supported: a claim that the excerpts cover is only partly supported, or close but not exact;
- not_supported: at least one claim is contradicted by the excerpts, or the excerpts of that document clearly show it
  does not say this."""
    judge = make_judge(
        name="groundedness",
        instructions="""You verify an answer of an assistant on aerospace quality documentation. {{ inputs }} holds the
conversation (the last user message is the question) and the evidence; {{ outputs }} is the answer.
List the answer's key factual claims (values, thresholds, deadlines, roles, steps, document identities, definitions,
compliance statements), ignoring greetings, generic advice and questions to the user. In a closing table of sources,
check the document codes and titles, but not the status or version labels ("current", "Courant", "—"): they are not
claims to verify, unless the evidence shows the document is cancelled, replaced or obsolete, which contradicts them.
Judge meaning, not wording: a faithful paraphrase, a summary, a heading or an abbreviated title is supported. Lower the
verdict only for a claim that matters to the user (value, rule, role, scope, condition, document identity) and is
wrong, overstated or only partly right.
""" + scope + """
Write the rationale in English and name the unsupported claims, if any.""",
        feedback_value_type=Literal["supported", "partially_supported", "not_supported"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"messages": inputs["messages"], "evidence": evidence}, outputs=outputs)


@scorer(name="missed_answer", description="yes when the answer says the information is not available, or leaves a part "
                                          "unanswered, while its evidence contains it.")
def missed_answer(inputs, outputs, trace):
    """yes when the answer says the information is not available (or leaves a part unanswered) while its evidence
    (RETRIEVER step) contains it. No assessment without evidence."""
    from typing import Literal

    from mlflow.entities import SpanType
    from mlflow.genai.judges import make_judge

    docs = [d if isinstance(d, dict) else d.to_dict()
            for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    if not docs:
        return None
    evidence = "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri')}] {d.get('page_content')}" for d in docs)
    judge = make_judge(
        name="missed_answer",
        instructions="""{{ inputs }} holds a conversation with an assistant on quality documentation (the last user
message is the question) and passages of quality documents; {{ outputs }} is the assistant's answer. Return yes if the
answer says the information is not available, or leaves a part of the question unanswered, while the passages DO
contain that information; otherwise return no. Asking the user to clarify a genuinely ambiguous request (a word or two
that could refer to many documents) is not a miss; leaving out a detail that the question does not ask for is not a
miss either. In the rationale (English), state what was missed, if anything.""",
        feedback_value_type=Literal["yes", "no"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"messages": inputs["messages"], "evidence": evidence}, outputs=outputs)


@scorer(name="retrieval_quality", description="Whether the assistant's retrieval found what the question needs: "
                                              "sufficient, retrieval_miss (an independent search of the index finds "
                                              "more), documentation_gap, not_applicable.")
def retrieval_quality(inputs, trace):
    """Separates retrieval errors from generation errors. A judge reads the passages the assistant retrieved; when they
    do not fully answer the question, a second judge reads an independent search of the whole index (corpus_search
    step). retrieval_miss: the index holds more than the assistant retrieved · documentation_gap: it does not.
    No assessment when the assistant's retrieval is not in its trace."""
    from typing import Literal

    from mlflow.entities import Feedback, SpanType
    from mlflow.genai.judges import make_judge

    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    if context.get("evidence_source") != "assistant_retrieval":
        return None
    retrieved = [d if isinstance(d, dict) else d.to_dict()
                 for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    searched = [d for s in trace.search_spans(name="corpus_search") for d in (s.outputs or []) if isinstance(d, dict)]

    def passages(docs):
        return "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri') or d.get('doc_uri')}] {d.get('page_content')}"
                              for d in docs) or "(no passage)"

    judge = make_judge(
        name="retrieval_quality",
        instructions="""{{ inputs }} holds a conversation with an assistant on the quality documentation of an aerospace
manufacturer (the last user message is the question) and passages of that documentation. Do the passages contain the
information needed to answer the question?
- full: everything the question asks is in the passages;
- partial: only part of it;
- none: nothing relevant;
- not_applicable: the question needs no documentation (greeting, thanks, out-of-scope request).
When the last message asks to rework the previous answer ("remove document X", "shorter", "in English"), judge the
passages against the question that answer addressed.
Write the rationale in English, in one or two sentences, naming what is missing, if anything.""",
        feedback_value_type=Literal["full", "partial", "none", "not_applicable"],
        model=trace.info.tags.get("judge_model") or None)
    first = judge(inputs={"messages": inputs["messages"], "passages": passages(retrieved)})
    tokens = ["mlflow.assessment.judgeInputTokens", "mlflow.assessment.judgeOutputTokens"]
    if first.error:
        return first
    if first.value in ("full", "not_applicable"):
        return Feedback(value="sufficient" if first.value == "full" else "not_applicable", rationale=first.rationale,
                        source=first.source, metadata=first.metadata)
    second = judge(inputs={"messages": inputs["messages"], "passages": passages(searched)})
    if second.error:
        return second
    rank = {"none": 0, "partial": 1, "full": 2}
    missed = rank.get(second.value, 0) > rank.get(first.value, 0)
    metadata = {k: sum(int((f.metadata or {}).get(k) or 0) for f in (first, second)) for k in tokens}
    return Feedback(value="retrieval_miss" if missed else "documentation_gap", source=first.source, metadata=metadata,
                    rationale=f"Assistant's passages: {first.value} ({first.rationale}) · Independent search of the "
                              f"index: {second.value} ({second.rationale})")


@scorer(name="compliance_claim", description="On questions asking whether the company meets a requirement: the answer "
                                             "concludes on compliance only as far as its evidence establishes it "
                                             "(evidence_based / unsupported_compliance_claim / not_applicable).")
def compliance_claim(inputs, outputs, trace):
    """Compliance-matrix questions ("does the company meet requirement X, which document proves it?", "same for Y"):
    unsupported_compliance_claim when the answer asserts compliance, or presents documents as proof, beyond what its
    evidence (RETRIEVER step) establishes."""
    from typing import Literal

    from mlflow.entities import SpanType
    from mlflow.genai.judges import make_judge

    docs = [d if isinstance(d, dict) else d.to_dict()
            for s in trace.search_spans(span_type=SpanType.RETRIEVER) for d in (s.outputs or [])]
    evidence = "\n---\n".join(f"[{(d.get('metadata') or {}).get('doc_uri')}] {d.get('page_content')}"
                              for d in docs) or "(no passage)"
    judge = make_judge(
        name="compliance_claim",
        instructions="""{{ inputs }} holds a conversation with an assistant on the quality documentation of an aerospace
manufacturer (the last user message is the question) and the documentation passages behind the answer; {{ outputs }}
is the answer. Apply this judge only when the question asks whether the company complies with, or meets, a requirement
(customer requirement, standard, line of a compliance matrix, "same for <requirement>"); otherwise return
not_applicable.
- evidence_based: the answer names the internal documents that address the requirement, says what they state, and
  concludes on compliance only as far as those statements establish it (or says that the evidence is partial or
  missing);
- unsupported_compliance_claim: the answer asserts compliance (or non-compliance) that the passages do not establish,
  or presents documents as proof when they do not address the requirement.
Write the rationale in English, in one or two sentences.""",
        feedback_value_type=Literal["evidence_based", "unsupported_compliance_claim", "not_applicable"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"messages": inputs["messages"], "evidence": evidence}, outputs=outputs)


@scorer(name="reference_integrity", description="Every document code cited in the answer exists (resolved typos and "
                                                "documents outside the corpus are accepted).")
def reference_integrity(trace):
    """True when every cited document code exists: resolved typos (zero padding) and documents outside the corpus
    mentioned in the excerpts are accepted; codes found nowhere are reported as unverified."""
    from mlflow.entities import Feedback

    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    unverified = context.get("unverified_refs") or []
    notes = []
    if context.get("approximate_refs"):
        notes.append(f"resolved: {context['approximate_refs']}")
    if context.get("unindexed_refs"):
        notes.append(f"outside the corpus: {context['unindexed_refs']}")
    if unverified:
        notes.append(f"unverified: {unverified}")
    return Feedback(value=not unverified, rationale="; ".join(notes) or "all cited codes exist")


SHARED_TRACE_SCORERS = [groundedness, missed_answer, retrieval_quality, compliance_claim, reference_integrity]


# ── Trace steps read by the scorers ──
def record_answer_context(context: dict):
    """Variable-length data of a turn or case (document lists, next user message, errors), stored as the trace step
    answer_context: trace tags are limited in length and a tag that is too long makes the whole trace fail."""
    with mlflow.start_span(name="answer_context", span_type="UNKNOWN") as span:
        span.set_outputs(context)


def answer_context(trace) -> dict:
    return next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})


# ── Evidence when the assistant's trace does not expose its retrieval: excerpts of the documents it cites ──
# Every document the answer relies on is checked, with no limit on the number of documents or excerpts and no
# truncation: each document is searched with the question, with every passage of the answer that cites it, and with
# the passages the assistant quoted from it (citation link fragments and footnotes).
_MARKER = re.compile(r"⟦\s*(\d+)\s*⟧")
_FOOTNOTE_REF = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_FOOTNOTE_DEF = re.compile(r"^\s*\[\^([^\]\s]+)\]:(.*)$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_URL = re.compile(r"https?://[^\s)\]>\"'\\]+")
_CODE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,28}")
MIN_QUERY_CHARS = 15


def quoted_passage(url: str) -> str:
    """Passage of a citation link's text fragment (#:~:text=[prefix-,]start[,end][,-suffix]), decoded."""
    from urllib.parse import unquote

    fragment = url.split("#:~:text=", 1)[1] if "#:~:text=" in url else ""
    parts = [p for p in fragment.split("&")[0].split(",") if p and not p.endswith("-") and not p.startswith("-")]
    return " … ".join(unquote(p) for p in parts).strip()


def _search_text(text: str) -> str:
    """Answer passage as a search query: links, citation markers and Markdown syntax removed."""
    text = _MD_LINK.sub(lambda m: m.group(1), str(text))
    text = _URL.sub(" ", _FOOTNOTE_REF.sub(" ", _MARKER.sub(" ", text)))
    return re.sub(r"\s+", " ", re.sub(r"[#|>]+|-{3,}", " ", re.sub(r"[*`]+", "", text))).strip()


def evidence_queries(question: str, raw_answer: str, documents: list, link_text: str = "") -> dict:
    """Search queries of every document the answer relies on: {cited code: [queries]}.
    documents: (code, citation number or None) pairs, e.g. the assistant's sources (number n of the ⟦n⟧ markers) and
    the codes written in the answer. Queries of a document: the question; each line of the answer that cites it
    (⟦n⟧ marker, footnote pointing to it, or its code written in the line); each passage the assistant quoted from it
    (text fragment of a citation link, footnote text). link_text: other text holding citation links (raw response)."""
    raw_answer = str(raw_answer or "")
    names, numbers = {}, {}
    for code, number in documents:
        key = base_ref(code)
        if len(key) >= 4 and key in REFS_BY_BASE:
            names.setdefault(key, str(code).strip())
            if number is not None:
                numbers.setdefault(str(number), set()).add(key)
    queries = {key: [question] for key in names}

    def keys_in(text: str) -> set:
        return {base_ref(t) for t in _CODE_TOKEN.findall(text)} & set(names)

    def add(keys, text):
        text = _search_text(text)
        if len(text) >= MIN_QUERY_CHARS:
            for key in keys:
                if text not in queries[key]:
                    queries[key].append(text)

    lines, footnotes = [], {}
    for line in raw_answer.splitlines():
        m = _FOOTNOTE_DEF.match(line)
        if m:
            footnotes[m.group(1)] = keys_in(m.group(2))
            add(footnotes[m.group(1)], m.group(2))              # passage quoted in the footnote
        else:
            lines.append(line)
    for line in lines:
        keys = keys_in(line) | {k for n in _MARKER.findall(line) for k in numbers.get(n, ())} \
               | {k for f in _FOOTNOTE_REF.findall(line) for k in footnotes.get(f, ())}
        add(keys, line)
    for label, url in _MD_LINK.findall(raw_answer + "\n" + str(link_text or "")):
        add(keys_in(label) | keys_in(url.split("#")[0]), quoted_passage(url))
    for url in _URL.findall(str(link_text or "")):
        add(keys_in(url.split("#")[0]), quoted_passage(url))
    return {names[key]: q for key, q in queries.items()}


@mlflow.trace(name="cited_document_excerpts", span_type="RETRIEVER")
def cited_document_excerpts(queries_by_document: dict) -> list:
    """Excerpts of every document the answer relies on, grouped by document: for each query of the document (see
    evidence_queries), the EXCERPTS_PER_QUERY most related chunks of that document and its language variants (Vector
    Search, hybrid), without duplicates and untruncated. Shown as a RETRIEVER step so that the retrieval judges can use
    them and they are readable in the trace."""
    excerpts, seen = [], set()
    for code, queries in queries_by_document.items():
        variants = sorted(REFS_BY_BASE.get(base_ref(code), ()))
        for query in queries if variants else ():
            res = w.vector_search_indexes.query_index(
                index_name=VS_INDEX, columns=VS_COLUMNS, query_text=query[:2000], query_type="HYBRID",
                num_results=EXCERPTS_PER_QUERY, filters_json=json.dumps({"REF": variants}))
            cols = [c.name for c in res.manifest.columns]
            for r in (dict(zip(cols, row)) for row in ((res.result.data_array if res.result else None) or [])):
                text = str(r.get("chunk_text") or "")
                if text and (r.get("REF"), text) not in seen:
                    seen.add((r.get("REF"), text))
                    excerpts.append(Document(id=f"{r.get('REF')}#{len(excerpts)}", page_content=text,
                                             metadata={"doc_uri": r.get("REF"),
                                                       "section": str(r.get("semantic_headers") or "")[:200]}))
    return excerpts

# ── What the assistant read: the retrieval steps of its own trace ──
_SOURCE_HEADER = re.compile(r"\[Source:\s*([^|\]\n]+)")


def trace_spans(trace) -> list:
    """Spans of an MLflow trace as {name, type, inputs, outputs}, from a Trace object or from its JSON form (the trace
    returned by a serving endpoint called with databricks_options.return_trace)."""
    data = trace.to_dict() if hasattr(trace, "to_dict") else (trace or {})
    spans = []
    for s in ((data.get("data") or {}).get("spans") or []) if isinstance(data, dict) else []:
        attributes = s.get("attributes") or {}

        def attribute(key):
            value = attributes.get(key)
            try:
                return json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                return value

        spans.append({"name": s.get("name"), "type": str(s.get("span_type") or attribute("mlflow.spanType") or "").upper(),
                      "inputs": s.get("inputs", attribute("mlflow.spanInputs")),
                      "outputs": s.get("outputs", attribute("mlflow.spanOutputs"))})
    return spans


def retrieved_passages(trace) -> tuple:
    """(number of retrieval steps, passages) of the assistant's trace: every document returned by its RETRIEVER spans,
    in order, without duplicates, as Documents whose doc_uri is the document code."""
    steps, passages, seen = 0, [], set()
    for span in trace_spans(trace):
        items = span["outputs"]
        if isinstance(items, dict):
            items = next((v for v in items.values() if isinstance(v, list)), [])
        items = [i for i in (items if isinstance(items, list) else []) if isinstance(i, dict)]
        is_documents = bool(items) and all(any(k in i for k in ("page_content", "content", "chunk_text")) for i in items)
        if span["type"] != "RETRIEVER" and not is_documents:
            continue
        steps += 1
        for item in items:
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            text = str(item.get("page_content") or item.get("content") or item.get("chunk_text") or item.get("text") or "")
            header = _SOURCE_HEADER.search(text)
            uri = str(meta.get("doc_uri") or item.get("doc_uri") or "")
            in_uri = re.search(r"[?&]ref=([A-Za-z0-9_.\-]+)", uri)
            doc = (meta.get("REF") or item.get("REF") or (header.group(1).strip() if header else None)
                   or (in_uri.group(1) if in_uri else None) or uri or meta.get("title") or "")
            if text and (doc, text) not in seen:
                seen.add((doc, text))
                passages.append(Document(id=f"{doc}#{len(passages)}", page_content=text,
                                         metadata={"doc_uri": str(doc), "step": str(span["name"])}))
    return steps, passages


def record_assistant_retrieval(assistant_trace_id: str, passages: list):
    """RETRIEVER step holding the passages the assistant retrieved, copied from its own trace."""
    with mlflow.start_span(name="assistant_retrieval", span_type="RETRIEVER") as span:
        span.set_inputs({"assistant_trace_id": assistant_trace_id})
        span.set_outputs(passages)


@mlflow.trace(name="corpus_search", span_type="TOOL")
def corpus_search(query: str) -> list:
    """Independent hybrid search of the whole index with the question: shows whether the documentation holds what the
    assistant's retrieval did not find (read by retrieval_quality)."""
    res = w.vector_search_indexes.query_index(index_name=VS_INDEX, columns=VS_COLUMNS, query_text=query[:2000],
                                              query_type="HYBRID", num_results=CORPUS_SEARCH_RESULTS)
    cols = [c.name for c in res.manifest.columns]
    rows = [dict(zip(cols, row)) for row in ((res.result.data_array if res.result else None) or [])]
    return [{"doc_uri": r.get("REF"), "page_content": str(r.get("chunk_text") or "")} for r in rows]


def search_query(messages: list) -> str:
    """Question used by the independent search: the last two user messages, so that a follow-up ("same for requirement
    X", "remove document Y") is searched with the question it refers to."""
    return "\n".join([m["content"] for m in messages if m["role"] == "user"][-2:])

# ── Numeric form of a verdict: 1 = pass, 0 = fail, 0.5 = partial; counts and durations as is; NULL for labels ──
SCORE_VALUES = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "full": 1.0, "partial": 0.5, "none": 0.0,
                "supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0,
                "no_contradiction": 1.0, "contradiction": 0.0, "correct_refusal": 1.0, "answered_anyway": 0.0,
                "good": 1.0, "acceptable": 0.5, "bad": 0.0, "up": 1.0, "down": 0.0,
                "sufficient": 1.0, "retrieval_miss": 0.0, "evidence_based": 1.0, "unsupported_compliance_claim": 0.0}
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
    """Name, raw value, rationale, source type, error and judge tokens (as reported by the judge model) of an MLflow
    assessment."""
    value = getattr(a, "value", None)
    fb = getattr(a, "feedback", None)
    err = getattr(fb, "error", None) if fb is not None else None
    meta = getattr(a, "metadata", None) or {}
    tokens = lambda key: int(meta[key]) if str(meta.get(key) or "").isdigit() else None
    return {"name": a.name, "value": getattr(value, "value", value), "rationale": getattr(a, "rationale", None),
            "source_type": str(getattr(getattr(a, "source", None), "source_type", "") or "").split(".")[-1].upper(),
            "error": (getattr(err, "error_message", None) or str(err))[:1000] if err else None,
            "input_tokens": tokens("mlflow.assessment.judgeInputTokens"),
            "output_tokens": tokens("mlflow.assessment.judgeOutputTokens")}


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


def source_citations(sources_json) -> list:
    """Documents listed by the assistant in sources_json, with their citation number: [(code, n), ...].
    n is the number of the ⟦n⟧ markers in the answer; None for a document returned but not cited."""
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
                out.append((str(ref).strip(), s.get("n")))
    return out


def source_refs(sources_json) -> list:
    """Document codes listed by the assistant in sources_json."""
    return list(dict.fromkeys(ref for ref, _ in source_citations(sources_json)))


REFS_BY_BASE, EXACT_REFS = {}, set()
CHUNK_CHARS = DEFAULT_CHUNK_CHARS
try:
    _src = REF_SOURCE_TABLE or w.vector_search_indexes.get_index(VS_INDEX).delta_sync_index_spec.source_table
    for r in spark.table(_src).select("REF").distinct().collect():
        if r.REF:
            REFS_BY_BASE.setdefault(base_ref(r.REF), set()).add(r.REF)
            EXACT_REFS.add(exact_ref(r.REF))
    print(f"Document index: {len(REFS_BY_BASE)} documents ({_src})")
    CHUNK_CHARS = float(spark.sql(f"SELECT AVG(LENGTH(chunk_text)) AS n FROM {_src}").collect()[0]["n"] or CHUNK_CHARS)
except Exception as e:
    if not REFS_BY_BASE:
        print(f"⚠️ Document index unavailable — excerpts, groundedness and reference checks disabled: {str(e)[:200]}")
print(f"Mean chunk size: {CHUNK_CHARS:.0f} characters (cost and rate estimates)")


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

# DBTITLE 1,Production scorers — question type, answer type, sampled safety, user reaction, citation count
INTENTS = ["definition_acronym", "document_lookup", "procedure_howto", "rule_requirement", "requirement_compliance",
           "comparison_multi_doc", "link_or_navigation", "person_or_org", "chitchat_or_meta", "out_of_scope"]
ANSWER_TYPES = ["answered_full", "answered_partial", "not_found", "out_of_scope_refusal", "clarification_request",
                "error_or_empty"]
COMPLETENESS = {"answered_full": "full", "answered_partial": "partial"}   # other answer types: not_applicable
SAFETY_SAMPLE_RATE = 0.10              # share of the turns checked by the safety judge (stable sample on message_id)

PRODUCTION_JUDGES = {
    "question_intent": (
        "What the last user message asks for.", Literal[tuple(INTENTS)], CONTEXT + """
Classify what the last user message asks for:
definition_acronym; document_lookup (find a document or template); procedure_howto; rule_requirement (rule, threshold,
deadline, responsibility); requirement_compliance (whether the company complies with a customer or standard requirement,
and which internal documents demonstrate it); comparison_multi_doc; link_or_navigation; person_or_org (a person, a team,
an organisation); chitchat_or_meta; out_of_scope."""),
    "answer_type": (
        "Kind of answer, and how completely it covers the question.", Literal[tuple(ANSWER_TYPES)], CONTEXT + """
Classify the answer:
- answered_full: answers every part of the question;
- answered_partial: answers only some parts, or stays vague on a part that was asked;
- not_found: says the documentation does not contain the information;
- out_of_scope_refusal: declines an off-topic request;
- clarification_request: asks the user to clarify instead of answering;
- error_or_empty: technical error message, or empty or truncated answer. An answer that addresses another question
  is still classified by what it does (answered_full, answered_partial…): relevance is judged separately.
Classify by how much of the question the answer covers, not by the correctness of its facts or the strength of its
evidence, which are judged separately."""),
}


@scorer(name="user_reaction", description="Implicit feedback carried by the user's next message: no_next_turn, moves_on, "
                                          "follow_up, rephrase_same_question, correction_or_complaint.")
def user_reaction(inputs, outputs, trace):
    """Classifies the user's next message (answer_context step of the trace, kept out of the inputs so that the other
    judges only see the conversation up to the question). Free (no judge call) when the user wrote nothing after it."""
    from typing import Literal

    from mlflow.entities import Feedback
    from mlflow.genai.judges import make_judge

    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    next_message = context.get("next_user_message")
    if not next_message:
        return Feedback(value="no_next_turn", rationale="The user wrote nothing after this answer.")
    question = next((m["content"] for m in reversed(inputs["messages"]) if m["role"] == "user"), "")
    judge = make_judge(
        name="user_reaction",
        instructions="""{{ inputs }} holds a user's question to an assistant on quality documentation and the message the
user wrote right after the assistant's answer; {{ outputs }} is that answer. Classify the user's next message:
moves_on (new unrelated question, or thanks); follow_up (natural continuation of the topic); rephrase_same_question
(asks the same thing again: the answer did not help); correction_or_complaint (says the answer is wrong, incomplete or
unhelpful). Write the rationale in English, in one sentence.""",
        feedback_value_type=Literal["moves_on", "follow_up", "rephrase_same_question", "correction_or_complaint"],
        model=trace.info.tags.get("judge_model") or None)
    return judge(inputs={"question": question, "next_user_message": next_message}, outputs=outputs)


@scorer(name="citation_count", description="Number of documents the assistant listed as sources.")
def citation_count(trace):
    """Number of documents the assistant listed as sources."""
    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    return len(context.get("source_refs") or [])


@scorer(name="safety", description="No harmful, offensive or leaked content (built-in judge), on a stable 10% sample "
                                   "of the turns.")
def safety(outputs, trace):
    """Built-in safety judge on the sampled turns only (trace tag safety_sample); no assessment otherwise."""
    from mlflow.genai.scorers import Safety

    if (trace.info.tags or {}).get("safety_sample") != "true":
        return None
    return Safety(model=trace.info.tags.get("judge_model") or None)(outputs=outputs)


def build_scorers(model):
    """(LLM judges called for every turn, judges called when applicable, code scorers)."""
    kw = {"model": model} if model else {}
    every_turn = [make_judge(name=n, description=d, feedback_value_type=t, instructions=i, **kw)
                  for n, (d, t, i) in PRODUCTION_JUDGES.items()] + shared_llm_judges(model)
    return (every_turn, [user_reaction, groundedness, missed_answer, retrieval_quality, compliance_claim, safety],
            [reference_integrity, citation_count])

# COMMAND ----------

# DBTITLE 1,Turn verdict — actionable rules on top of the scorers, and the stage at fault (retrieval or generation)
# Stage at fault of each failure reason: retrieval = the assistant's search did not bring what the index holds;
# generation = the model misused what it retrieved (claims not in its passages, information it had but left out).
REASON_STAGE = {"retrieval_miss": "retrieval",
                "unsupported_claims": "generation", "partially_supported_claims": "generation",
                "unsupported_compliance_claim": "generation", "missed_answer_in_sources": "generation",
                "missed_information": "generation", "wrongful_refusal": "generation", "off_topic": "generation",
                "unverified_reference": "generation", "language_mismatch": "generation", "no_citation": "generation"}


def turn_verdict(v: dict) -> tuple:
    """(verdict, failure_reasons, error_source) from the scorer values. bad = the user was badly served; acceptable =
    minor issue; error_source = retrieval, generation or retrieval_and_generation, from the reasons of the verdict (a
    retrieval miss always counts: the index held what the assistant did not retrieve)."""
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
        if at in ("not_found", "out_of_scope_refusal", "clarification_request"):
            bad.append("missed_answer_in_sources")        # "not found" although the cited documents contain it
        else:
            warn.append("missed_information")             # answered, but left out details the documents contain
    if v.get("groundedness") == "not_supported":
        bad.append("unsupported_claims")
    elif v.get("groundedness") == "partially_supported":
        warn.append("partially_supported_claims")
    if v.get("compliance_claim") == "unsupported_compliance_claim":
        bad.append("unsupported_compliance_claim")
    if v.get("retrieval_quality") == "retrieval_miss":
        if at in ("not_found", "answered_partial", "clarification_request", "out_of_scope_refusal"):
            bad.append("retrieval_miss")                  # the index holds what the user did not get
        else:
            warn.append("retrieval_miss")
    if at == "answered_partial":
        warn.append("incomplete")
    if v.get("reference_integrity") is False:
        warn.append("unverified_reference")
    if v.get("language_match") == "no":
        warn.append("language_mismatch")
    if at in ("answered_full", "answered_partial") and intent not in ("out_of_scope", "chitchat_or_meta") \
            and not v.get("citation_count"):
        warn.append("no_citation")
    if v.get("user_reaction") == "correction_or_complaint":
        warn.append("user_complaint")
    elif v.get("user_reaction") == "rephrase_same_question":
        warn.append("user_rephrased")
    stages = {REASON_STAGE[r] for r in (bad or warn) if r in REASON_STAGE}
    if "retrieval_miss" in warn:                          # what the assistant got wrong, the index held
        stages.add("retrieval")
    source = "retrieval_and_generation" if len(stages) > 1 else next(iter(stages), None)
    return ("bad" if bad else "acceptable" if warn else "good"), bad + warn, source

# COMMAND ----------

# DBTITLE 1,Judge check and registration — the run stops if the judge model does not answer
import inspect

JUDGE_MODEL = f"databricks:/{JUDGE_ENDPOINT}" if JUDGE_ENDPOINT else None
_sample = {"inputs": {"messages": [{"role": "user", "content": "What is the retention period of inspection records?"}]},
           "outputs": "According to **QP-1457**, inspection records are kept for 10 years."}


if not DRY_RUN:
    try:
        _error = getattr(build_scorers(JUDGE_MODEL)[0][0](**_sample), "error", None)
    except Exception as e:
        _error = e
    if _error:
        raise RuntimeError(f"Judge model {JUDGE_MODEL or 'Databricks-managed'} does not answer (check the endpoint name "
                           f"in Serving): {str(_error)[:300]}")
LLM_JUDGES, TRACE_JUDGES, CODE_SCORERS = build_scorers(JUDGE_MODEL)
SCORERS = LLM_JUDGES + TRACE_JUDGES + CODE_SCORERS

# Fingerprint of everything that determines a score: scorers, judge model, verdict rules, excerpts, history window
JUDGE_CONFIG_ID = scorers_config_id(SCORERS, JUDGE_MODEL, inspect.getsource(turn_verdict), json.dumps(REASON_STAGE),
                                    inspect.getsource(retrieved_passages), inspect.getsource(evidence_queries),
                                    EXCERPTS_PER_QUERY, CORPUS_SEARCH_RESULTS, CHAT_HISTORY_LIMIT)
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
if spark.catalog.tableExists(SCORES_TABLE) and not RESET_OUTPUTS:
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

# DBTITLE 1,Replayed turns — evaluation records and the traced replay (answer, what the assistant retrieved, index search)

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


def for_judges(messages: list) -> list:
    """Conversation as given to the judges: the messages the assistant saw, whole, citation text fragments removed."""
    return [{"role": m["role"], "content": clean_answer(m["content"])} for m in messages]


def last_user_question(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def in_sample(key, rate: float, salt: str = "") -> bool:
    """Stable pseudo-random sample: the same keys are selected at every run."""
    return int(hashlib.md5(f"{salt}{key}".encode()).hexdigest()[:8], 16) % 10000 < rate * 10000


def _text(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


TURNS = {}
for _, row in pdf_pairs.iterrows():
    thread = for_judges(trim_history(to_messages(row["prior_messages"])))
    raw_answer = str(row["answer"] or "")
    question = last_user_question(thread)
    cited = sorted(code_like(raw_answer))
    TURNS[str(row["message_id"])] = {
        "row": row, "thread": thread, "question": question, "answer": clean_answer(raw_answer),
        "next_user_message": _text(row.get("next_user_message")),
        "source_refs": source_refs(row["sources_json"]), "cited_refs": cited,
        "evidence_queries": evidence_queries(question, raw_answer,
                                             source_citations(row["sources_json"]) + [(c, None) for c in cited]),
    }
RECORDS = {mid: {"inputs": {"messages": t["thread"], "message_id": mid}} for mid, t in TURNS.items()}
TRACE_IDS = {}                          # message_id → scoring trace id
ASSISTANT_TRACES = {}                   # trace_id of chat_messages → (assistant trace or None, error)


def assistant_trace(trace_id) -> tuple:
    """(the assistant's own MLflow trace of a turn, error): looked up by the trace_id stored in chat_messages, as is and
    in the MLflow form tr-<32 hex>."""
    raw = str(trace_id or "").strip()
    if not raw:
        return None, "no trace id in chat_messages"
    if raw not in ASSISTANT_TRACES:
        errors = []
        for candidate in dict.fromkeys([raw, raw if raw.startswith("tr-") else f"tr-{raw.replace('-', '')}"]):
            try:
                found = mlflow.get_trace(candidate)
            except Exception as e:
                found = None
                errors.append(f"{candidate}: {str(e)[:150]}")
            if found is not None:
                ASSISTANT_TRACES[raw] = (found, None)
                break
        else:
            ASSISTANT_TRACES[raw] = (None, " | ".join(errors) or f"{raw}: trace not found")
    return ASSISTANT_TRACES[raw]


@mlflow.trace(name="qualibot_turn", span_type="AGENT")
def replay_turn(messages, message_id):
    """Returns the answer stored in chat_messages; the trace carries the turn's identifiers and cited documents."""
    t = TURNS[message_id]
    row = t["row"]
    TRACE_IDS[message_id] = mlflow.get_current_active_span().trace_id
    # Short identifiers only in the tags; document lists and the next user message go to the answer_context step
    mlflow.update_current_trace(
        tags={"message_id": message_id, "endpoint": str(row["endpoint_name"]), "division": str(row["division"]),
              "agent_trace_id": _text(row.get("trace_id")) or "", "judge_model": JUDGE_MODEL or "",
              "judge_config_id": JUDGE_CONFIG_ID,
              "safety_sample": str(in_sample(message_id, SAFETY_SAMPLE_RATE, "safety")).lower()},
        metadata={"mlflow.trace.session": str(row["session_id"])})
    kt, kt_error = assistant_trace(row.get("trace_id"))
    steps, passages = retrieved_passages(kt) if kt is not None else (0, [])
    context = {"source_refs": t["source_refs"], "cited_refs": t["cited_refs"],
               "next_user_message": t["next_user_message"], "retrieval_error": None,
               "assistant_trace_error": kt_error, "assistant_retrieval_steps": steps}
    docs = []
    try:
        # Evidence: what the assistant retrieved; excerpts of the cited documents only when its trace does not show it
        if steps:
            context["evidence_source"] = "assistant_retrieval"
            docs = passages
            record_assistant_retrieval(kt.info.trace_id, passages)
            if REFS_BY_BASE:
                corpus_search(search_query(t["thread"]))
        elif t["evidence_queries"]:
            context["evidence_source"] = "cited_document_excerpts"
            docs = cited_document_excerpts(t["evidence_queries"])
        else:
            context["evidence_source"] = "none"
    except Exception as e:
        context["retrieval_error"] = str(e)[:500]
    classes = classify_refs(t["cited_refs"], "\n".join(d.page_content for d in docs))
    record_answer_context({**context, "evidence_refs": sorted({d.metadata["doc_uri"] for d in docs}),
                           "evidence_count": len(docs), "evidence_chars": sum(len(d.page_content) for d in docs),
                           **{f"{k}_refs": v for k, v in classes.items()}})
    return t["answer"]

# COMMAND ----------

# DBTITLE 1,Assistant traces — the passages the assistant retrieved, read from its own trace (sample check)
# The judges compare each answer with the passages the assistant retrieved. They are read from the assistant's MLflow
# trace (trace_id of chat_messages); when a trace shows no retrieval step, its span structure is printed below and the
# judges fall back on excerpts of the cited documents.
_sampled_trace_ids = [t["row"].get("trace_id") for t in TURNS.values() if _text(t["row"].get("trace_id"))][:5]
for tid in _sampled_trace_ids:
    kt, error = assistant_trace(tid)
    if kt is None:
        print(f"✗ {tid}: {error}")
        continue
    steps, passages = retrieved_passages(kt)
    print(f"✓ {tid} → {kt.info.trace_id}: {steps} retrieval step(s), {len(passages)} passage(s)"
          + (f", first from {passages[0].metadata['doc_uri']}" if passages else ""))
    if not steps:
        for span in trace_spans(kt):
            out = span["outputs"]
            first = out[0] if isinstance(out, list) and out else out
            keys = sorted(first)[:8] if isinstance(first, dict) else type(first).__name__
            print(f"    {span['type'] or '-':10} {str(span['name'])[:60]:60} outputs: {type(out).__name__} {keys}")
if not _sampled_trace_ids:
    print("No turn with a trace_id: the judges read excerpts of the cited documents.")

# COMMAND ----------

# DBTITLE 1,Records — one row per turn and one per turn × scorer; verdict and user vote attached to the trace
from mlflow.entities import AssessmentSource, AssessmentSourceType

RULE_SOURCE = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="turn_verdict_rules")
USER_SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="chat_user")
REQUIRED = {"answer_type", "relevance", "question_intent"}      # without them the verdict is not meaningful


def estimated_usage(t: dict) -> tuple:
    """(judge calls, input tokens, output tokens) of one turn, estimated from the prompt sizes before it is scored (dry
    run, batch planning). The evidence is taken as the chunks of every search of the turn (an upper bound)."""
    searches = max(sum(map(len, t["evidence_queries"].values())) * EXCERPTS_PER_QUERY, CORPUS_SEARCH_RESULTS)
    evidence_chars = searches * CHUNK_CHARS
    exchange = JUDGE_INSTRUCTIONS_CHARS + len(json.dumps(t["thread"], ensure_ascii=False)) + len(t["answer"])
    evidence_judges = 4.5 if t["evidence_queries"] or t["row"].get("trace_id") else 1    # incl. compliance_claim
    reaction = 1 if t["next_user_message"] else 0
    calls = len(LLM_JUDGES) + evidence_judges + reaction + SAFETY_SAMPLE_RATE
    chars = ((len(LLM_JUDGES) + reaction + SAFETY_SAMPLE_RATE) * exchange
             + evidence_judges * (exchange + evidence_chars))
    return calls, chars / CHARS_PER_TOKEN, calls * OUTPUT_TOKENS_PER_JUDGE_CALL


def measured_usage(assessments: list) -> tuple:
    """(judge calls, input tokens, output tokens) of a scored turn, as reported by the judge model."""
    judged = [a for a in assessments if a["source_type"] == "LLM_JUDGE"]
    return (len(judged), sum(a.get("input_tokens") or 0 for a in judged),
            sum(a.get("output_tokens") or 0 for a in judged))


def cost_usd(tokens_in, tokens_out) -> float:
    return (tokens_in * DBU_PER_M_INPUT + tokens_out * DBU_PER_M_OUTPUT) / 1e6 * USD_PER_DBU


def in_calibration_sample(message_id) -> bool:
    return in_sample(message_id, HUMAN_REVIEW_SAMPLE_RATE)


def _yes(v):
    return None if v is None else v == "yes"


def build_record(mid: str) -> dict:
    t = TURNS[mid]
    res = results.get(mid, {"trace_id": None, "context": {}, "assessments": []})
    row, context = t["row"], res["context"]
    ok_rows = [a for a in res["assessments"] if not a["error"] and a["value"] is not None]
    v = {a["name"]: a["value"] for a in ok_rows}
    why = {a["name"]: a["rationale"] for a in ok_rows}
    errors = [f"{a['name']}: {a['error'][:200]}" for a in res["assessments"] if a["error"]]
    if not res["trace_id"]:
        errors.append("turn not scored")
    if context.get("retrieval_error"):
        errors.append(f"vector_search: {context['retrieval_error']}")
    verdict, reasons, stage = turn_verdict(v) if REQUIRED <= set(v) else (None, ["judge_failed"], None)
    vote = _text(row.get("feedback_vote"))
    disagreement = (verdict == "good" and vote == "down") or (verdict == "bad" and vote == "up")
    review = bool(verdict) and bool(disagreement or in_calibration_sample(mid))
    calls, t_in, t_out = measured_usage(res["assessments"])
    if not t_in:
        calls, t_in, t_out = estimated_usage(t)
    ground = v.get("groundedness")
    return {
        "message_id": row["message_id"], "created_at": row["created_at"], "session_id": row["session_id"],
        "division": row["division"], "endpoint_name": row["endpoint_name"], "trace_id": _text(row.get("trace_id")),
        "scoring_trace_id": res["trace_id"], "mlflow_run_id": MLFLOW_RUN_ID,
        "user_question": t["question"], "thread_turn_count": len(t["thread"]), "answer": t["answer"],
        "next_user_message": t["next_user_message"],
        "citation_count": len(t["source_refs"]), "source_refs": t["source_refs"], "cited_refs": t["cited_refs"],
        "evidence_source": context.get("evidence_source"), "evidence_refs": context.get("evidence_refs") or [],
        "evidence_count": context.get("evidence_count"),
        "assistant_retrieval_steps": context.get("assistant_retrieval_steps"),
        "assistant_trace_error": context.get("assistant_trace_error"),
        "approximate_refs": context.get("approximate_refs") or [],
        "unindexed_refs": context.get("unindexed_refs") or [],
        "unverified_refs": context.get("unverified_refs") or [],
        "feedback_vote": vote, "feedback_comment": _text(row.get("feedback_comment")),
        "question_intent": v.get("question_intent"),
        "in_scope": None if not v.get("question_intent") else
                    ("no" if v["question_intent"] in ("out_of_scope", "chitchat_or_meta") else "yes"),
        "is_follow_up": sum(m["role"] == "user" for m in t["thread"]) > 1,
        "answer_type": v.get("answer_type"),
        "relevance__value": _yes(v.get("relevance")), "relevance__rationale": why.get("relevance"),
        "answer_type__rationale": why.get("answer_type"),
        "completeness_level": None if not v.get("answer_type") else COMPLETENESS.get(v["answer_type"], "not_applicable"),
        "completeness__value": None if not v.get("answer_type") else v["answer_type"] != "answered_partial",
        "language_match__value": _yes(v.get("language_match")), "language_match__rationale": why.get("language_match"),
        "safety__value": _yes(v.get("safety")), "safety__rationale": why.get("safety"),
        "groundedness_level": ground,
        "groundedness__value": None if ground is None else ground == "supported",
        "groundedness__rationale": why.get("groundedness"),
        "missed_answer": _yes(v.get("missed_answer")), "missed_answer_detail": why.get("missed_answer"),
        "retrieval_quality": v.get("retrieval_quality"), "retrieval_quality__rationale": why.get("retrieval_quality"),
        "compliance_claim": v.get("compliance_claim"), "compliance_claim__rationale": why.get("compliance_claim"),
        "next_turn_signal": v.get("user_reaction"), "next_turn_rationale": why.get("user_reaction"),
        "turn_verdict": verdict, "failure_reasons": reasons, "error_source": stage,
        "needs_human_review": review,
        "review_reason": "judge_vs_user_disagreement" if disagreement and review else "calibration_sample" if review else None,
        "golden_candidate": verdict == "bad" or vote == "down",
        "judge_model": JUDGE_MODEL or "databricks-managed", "judge_config_id": JUDGE_CONFIG_ID,
        "judge_errors": errors, "n_judge_calls": int(round(calls)),
        "judge_input_tokens": int(t_in), "judge_output_tokens": int(t_out),
        "judge_cost_usd": round(cost_usd(t_in, t_out), 6), "scored_at": RUN_TS,
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
             "value_numeric": numeric_value(a["name"], a["value"]), "rationale": a["rationale"], "error": a["error"],
             "judge_input_tokens": a.get("input_tokens"), "judge_output_tokens": a.get("output_tokens")}
            for a in items]


def score_batch(batch: list) -> tuple:
    """Reads the scores of a scored batch from its traces, builds its rows and attaches the verdict and the user's vote
    to each trace. Returns (turn rows, turn × scorer rows)."""
    for mid in batch:
        if mid in TRACE_IDS:
            tr = mlflow.get_trace(TRACE_IDS[mid])
            results[mid] = {"trace_id": tr.info.trace_id, "context": answer_context(tr),
                            "assessments": [assessment_row(a) for a in (tr.info.assessments or [])]}
    rows = [build_record(mid) for mid in batch]
    for r in rows:
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
    return rows, [a for mid, r in zip(batch, rows) for a in assessment_records(mid, r)]

# COMMAND ----------

# DBTITLE 1,Unity Catalog tables — documented schemas, rows replaced by key, run ledger
import pandas as pd
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
    ("answer", S, "Assistant answer, without citation text fragments"),
    ("next_user_message", S, "Message the user wrote after the answer, if any"),
    ("citation_count", I, "Documents listed by the assistant as sources"),
    ("source_refs", A, "Document codes listed by the assistant as sources"),
    ("cited_refs", A, "Document codes cited in the answer text"),
    ("evidence_source", S, "Evidence read by the judges: assistant_retrieval (passages the assistant retrieved, from its "
                           "trace), cited_document_excerpts (when its trace shows no retrieval) or none"),
    ("evidence_refs", A, "Documents of the evidence passages"),
    ("evidence_count", I, "Evidence passages read by the judges"),
    ("assistant_retrieval_steps", I, "Retrieval steps found in the assistant's trace (0: trace not found or without them)"),
    ("assistant_trace_error", S, "Why the assistant's trace could not be read, if so"),
    ("approximate_refs", A, "Cited codes resolved despite a typo (e.g. IN_APO_006 → IN_APO_0006)"),
    ("unindexed_refs", A, "Cited codes absent from the index but mentioned in the excerpts (documents outside the corpus)"),
    ("unverified_refs", A, "Cited codes found neither in the index nor in the excerpts (possibly invented)"),
    ("feedback_vote", S, "User vote on the answer: up, down or NULL"),
    ("feedback_comment", S, "User comment attached to the vote"),
    ("question_intent", S, "Judge label: kind of question (definition_acronym, document_lookup, requirement_compliance…)"),
    ("in_scope", S, "yes unless the intent is out_of_scope or chitchat_or_meta"),
    ("is_follow_up", B, "The conversation had earlier user messages"),
    ("answer_type", S, "Judge label: answered_full, answered_partial, not_found, out_of_scope_refusal, clarification_request, error_or_empty"),
    ("answer_type__rationale", S, "Rationale of the answer type judge"),
    ("relevance__value", B, "Judge: the answer addresses the question"),
    ("relevance__rationale", S, "Rationale of the relevance judge"),
    ("completeness_level", S, "From answer_type: full, partial, or not_applicable (refusal, not found, clarification)"),
    ("completeness__value", B, "The answer is not partial"),
    ("language_match__value", B, "Judge: the answer is in the language of the question"),
    ("language_match__rationale", S, "Rationale of the language judge"),
    ("safety__value", B, "Judge: no harmful content; NULL outside the 10% safety sample"),
    ("safety__rationale", S, "Rationale of the safety judge"),
    ("groundedness_level", S, "Judge: supported, partially_supported or not_supported; NULL when nothing could be checked"),
    ("groundedness__value", B, "groundedness_level is supported"),
    ("groundedness__rationale", S, "Rationale of the groundedness judge (names the unsupported claims)"),
    ("missed_answer", B, "Judge: the answer says 'not found' while the cited excerpts contain the information"),
    ("missed_answer_detail", S, "What was missed, according to the judge"),
    ("retrieval_quality", S, "sufficient: the assistant retrieved what the question needs · retrieval_miss: an independent "
                             "search of the index finds more · documentation_gap · not_applicable; NULL without the "
                             "assistant's retrieval"),
    ("retrieval_quality__rationale", S, "Rationale of the retrieval quality judges"),
    ("compliance_claim", S, "Compliance questions: evidence_based, unsupported_compliance_claim or not_applicable"),
    ("compliance_claim__rationale", S, "Rationale of the compliance claim judge"),
    ("next_turn_signal", S, "Judge label of the user's next message: no_next_turn, moves_on, follow_up, rephrase_same_question, correction_or_complaint"),
    ("next_turn_rationale", S, "Rationale of the user reaction judge"),
    ("turn_verdict", S, _VERDICT_DOC),
    ("failure_reasons", A, "Reasons of a bad or acceptable verdict (unsupported_claims, missed_answer_in_sources, …)"),
    ("error_source", S, "Stage at fault: retrieval, generation or retrieval_and_generation; NULL for a good verdict"),
    ("needs_human_review", B, "In the human review queue (judge/user disagreement or calibration sample)"),
    ("review_reason", S, "judge_vs_user_disagreement or calibration_sample"),
    ("golden_candidate", B, "Bad verdict or down vote: candidate case for the golden dataset"),
    ("judge_model", S, "Judge model used"),
    ("judge_config_id", S, "Fingerprint of the scorers, judge model and verdict rules; compare scores of the same configuration"),
    ("judge_errors", A, "Scorer errors of the turn"),
    ("n_judge_calls", I, "LLM judge assessments of the turn"),
    ("judge_input_tokens", I, "Input tokens of the judge calls, as reported by the judge model (estimated if not)"),
    ("judge_output_tokens", I, "Output tokens of the judge calls, as reported by the judge model (estimated if not)"),
    ("judge_cost_usd", D, "Judge cost of the turn (USD), from its tokens"),
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
    ("judge_input_tokens", I, "Input tokens of the judge call(s), as reported by the judge model"),
    ("judge_output_tokens", I, "Output tokens of the judge call(s), as reported by the judge model"),
]
ASSESSMENTS_SCHEMA = StructType([StructField(n, t) for n, t, _ in ASSESSMENTS_COLUMNS])
ASSESSMENTS_DOCS = {n: d for n, _, d in ASSESSMENTS_COLUMNS}

RUNS_COLUMNS = [
    ("run_ts", S, "Start time (UTC) of the scoring run"),
    ("mlflow_run_id", S, "MLflow run of the scoring run"),
    ("judge_model", S, "Judge model used"),
    ("judge_config_id", S, "Fingerprint of the scorers, judge model and verdict rules"),
    ("n_messages", I, "Turns scored"),
    ("n_judge_calls", I, "LLM judge assessments"),
    ("judge_input_tokens", I, "Input tokens of the judge calls"),
    ("judge_output_tokens", I, "Output tokens of the judge calls"),
    ("judge_cost_usd", D, "Judge cost (USD), from the tokens"),
    ("duration_s", D, "Duration of the run (seconds)"),
    ("n_judge_errors", I, "Turns with at least one scorer error"),
    ("good_rate", D, "Share of good verdicts"),
    ("bad_rate", D, "Share of bad verdicts"),
    ("refusal_rate", D, "Share of not_found and out_of_scope_refusal answers"),
    ("groundedness_rate", D, "Share of supported answers among those checked against excerpts"),
    ("grounding_coverage", D, "Share of answers checked against evidence"),
    ("assistant_retrieval_coverage", D, "Share of turns whose evidence is the assistant's own retrieval"),
    ("retrieval_miss_rate", D, "Share of retrieval_miss among the turns whose retrieval quality was judged"),
    ("judge_user_agreement", D, "Agreement between the verdict (bad / not bad) and the user votes"),
    ("n_voted", I, "Scored turns with a user vote"),
    ("n_needs_human_review", I, "Turns added to the human review queue"),
    ("n_turns_left", I, "Turns left for the next run when the time budget was reached"),
]
RUNS_SCHEMA = StructType([StructField(n, t) for n, t, _ in RUNS_COLUMNS])
RUNS_DOCS = {n: d for n, _, d in RUNS_COLUMNS}


_tables_ready = False


def write_turns(rows: list, assessment_rows: list):
    """Writes a batch of scored turns. The first write creates the tables; with reset_outputs it first drops them, so a
    run that fails before its first batch keeps the previous tables. The tables belong to the job identity."""
    global _tables_ready
    if not _tables_ready:
        if RESET_OUTPUTS:
            for table in (SCORES_TABLE, ASSESSMENTS_TABLE, SCORING_RUNS_TABLE):
                spark.sql(f"DROP TABLE IF EXISTS {table}")
            print(f"Output tables reset: {SCORES_TABLE}, {ASSESSMENTS_TABLE}, {SCORING_RUNS_TABLE}")
        ensure_table(SCORES_TABLE, SCORES_SCHEMA, "Qualibot production turns scored by LLM judges: one row per "
                     "assistant turn (labels, verdicts, rationales, references). Written by the quality scoring job.",
                     SCORES_DOCS)
        ensure_table(ASSESSMENTS_TABLE, ASSESSMENTS_SCHEMA, "Qualibot production scoring: one row per assistant turn "
                     "and scorer, including the rule-based turn_verdict and the user's vote.", ASSESSMENTS_DOCS)
        ensure_table(SCORING_RUNS_TABLE, RUNS_SCHEMA, "Qualibot production scoring runs: volumes, rates and "
                     "estimated cost.", RUNS_DOCS)
        _tables_ready = True
    replace_rows(SCORES_TABLE, SCORES_SCHEMA, rows, ["message_id"])
    replace_rows(ASSESSMENTS_TABLE, ASSESSMENTS_SCHEMA, assessment_rows, ["message_id"])


def rate(series, value=True):
    s = series.dropna()
    return round(float((s == value).mean()), 4) if len(s) else None


def write_run(df: pd.DataFrame, n_left: int) -> dict:
    """Writes (or updates) the run's row of the run ledger from the turns scored so far."""
    voted = df[df["feedback_vote"].isin(["up", "down"]) & df["turn_verdict"].notna()]
    agree = round(float(((voted["turn_verdict"] != "bad") == (voted["feedback_vote"] == "up")).mean()), 4) if len(voted) else None
    row = {
        "run_ts": RUN_TS, "mlflow_run_id": MLFLOW_RUN_ID, "judge_model": JUDGE_MODEL or "databricks-managed",
        "judge_config_id": JUDGE_CONFIG_ID, "n_messages": int(len(df)),
        "n_judge_calls": int(df["n_judge_calls"].sum()),
        "judge_input_tokens": int(df["judge_input_tokens"].sum()), "judge_output_tokens": int(df["judge_output_tokens"].sum()),
        "judge_cost_usd": round(float(df["judge_cost_usd"].sum()), 6),
        "duration_s": round(time.time() - t_start, 1),
        "n_judge_errors": int((df["judge_errors"].map(len) > 0).sum()),
        "good_rate": rate(df["turn_verdict"], "good"), "bad_rate": rate(df["turn_verdict"], "bad"),
        "refusal_rate": rate(df["answer_type"].isin(["not_found", "out_of_scope_refusal"]).where(df["answer_type"].notna())),
        "groundedness_rate": rate(df["groundedness__value"]),
        "grounding_coverage": rate(df["groundedness_level"].notna()),
        "assistant_retrieval_coverage": rate(df["evidence_source"] == "assistant_retrieval"),
        "retrieval_miss_rate": rate(df["retrieval_quality"].where(df["retrieval_quality"].isin(
            ["sufficient", "retrieval_miss", "documentation_gap"])), "retrieval_miss"),
        "judge_user_agreement": agree, "n_voted": int(len(voted)),
        "n_needs_human_review": int(df["needs_human_review"].sum()), "n_turns_left": int(n_left),
    }
    replace_rows(SCORING_RUNS_TABLE, RUNS_SCHEMA, [row], ["mlflow_run_id"])
    return row

# COMMAND ----------

# DBTITLE 1,Run — batches of turns scored with mlflow.genai.evaluate, paced on the judge model's rate limits
# Each batch holds about one minute of this job's share of the judge model's token limits, estimated from the prompt
# sizes and corrected by the tokens the judge model reports. Its scores are written before the next batch starts, so
# an interrupted run keeps what it scored. No batch starts after max_run_minutes: the turns left go to the next run.
from datetime import datetime, timezone

RUN_TS = datetime.now(timezone.utc).isoformat(timespec="seconds")
t_start = time.time()
MLFLOW_RUN_ID = None
results, records, assessment_rows, run_row = {}, [], [], None
calibration = [1.0, 1.0]               # measured / estimated tokens (input, output), updated after each batch
TOKENS_IN_PER_MINUTE = JUDGE_INPUT_TOKENS_PER_MINUTE * JUDGE_RATE_SHARE
TOKENS_OUT_PER_MINUTE = JUDGE_OUTPUT_TOKENS_PER_MINUTE * JUDGE_RATE_SHARE


def minutes_of_limits(usage: list) -> float:
    """Minutes of this job's share of the rate limits taken by a list of (calls, input tokens, output tokens)."""
    return max(sum(u[1] for u in usage) / TOKENS_IN_PER_MINUTE, sum(u[2] for u in usage) / TOKENS_OUT_PER_MINUTE)


def next_batch(pending: list) -> list:
    """First pending turns whose calibrated judge usage fits in one minute of the limits (at least one turn)."""
    batch, usage = [], []
    for mid in pending:
        calls, t_in, t_out = estimated_usage(TURNS[mid])
        usage.append((calls, t_in * calibration[0], t_out * calibration[1]))
        if batch and minutes_of_limits(usage) > 1:
            break
        batch.append(mid)
    return batch


def rate_limited(rows: list) -> bool:
    return any("429" in (a["error"] or "") or "rate limit" in (a["error"] or "").lower() for a in rows)


if not TURNS:
    print("Nothing new to score.")
elif DRY_RUN:
    usage = [estimated_usage(t) for t in TURNS.values()]
    calls, t_in, t_out = (sum(u[i] for u in usage) for i in range(3))
    print(f"DRY RUN (nothing is scored or written) · {len(TURNS)} turns · ~{calls:.0f} judge calls "
          f"({calls / len(TURNS):.1f} per turn) · at most ~{t_in / 1e6:.2f}M in / ~{t_out / 1e6:.2f}M out tokens · "
          f"at most ≈ ${cost_usd(t_in, t_out):.2f} · at most {minutes_of_limits(usage):.0f} min of judge rate limits "
          f"(budget per run: {MAX_RUN_MINUTES:.0f} min)")
else:
    pending, pace = list(TURNS), 1.0
    with mlflow.start_run(run_name=f"quality-scoring {RUN_TS}") as run:
        MLFLOW_RUN_ID = run.info.run_id
        mlflow.set_tags({"judge_model": JUDGE_MODEL or "databricks-managed", "judge_config_id": JUDGE_CONFIG_ID,
                         "source_table": SOURCE_TABLE})
        mlflow.log_params({"n_turns": len(TURNS), "test_limit": TEST_LIMIT, "excerpts_per_query": EXCERPTS_PER_QUERY,
                           "judge_rate_share": JUDGE_RATE_SHARE, "max_run_minutes": MAX_RUN_MINUTES})
        while pending and time.time() - t_start < MAX_RUN_MINUTES * 60:
            batch, b_start = next_batch(pending), time.time()
            pending = pending[len(batch):]
            mlflow.genai.evaluate(data=[RECORDS[mid] for mid in batch], predict_fn=replay_turn, scorers=SCORERS)
            rows, a_rows = score_batch(batch)
            write_turns(rows, a_rows)
            records += rows
            assessment_rows += a_rows
            run_row = write_run(pd.DataFrame(records), len(pending))
            # Pacing: the batch lasts as long as its measured usage takes of the limits; slower after a rejection
            pace = min(pace * 1.5, 4.0) if rate_limited(a_rows) else max(1.0, pace / 1.2)
            batch_usage = [(r["n_judge_calls"], r["judge_input_tokens"], r["judge_output_tokens"]) for r in rows]
            estimated = [estimated_usage(TURNS[mid]) for mid in batch]
            for i in (0, 1):
                measured, planned = sum(u[i + 1] for u in batch_usage), sum(e[i + 1] for e in estimated)
                if measured and planned:
                    calibration[i] = 0.5 * calibration[i] + 0.5 * measured / planned
            print(f"{len(records)}/{len(TURNS)} turns scored · {time.time() - t_start:.0f} s · batch of {len(batch)}: "
                  f"{sum(u[1] for u in batch_usage):,} in / {sum(u[2] for u in batch_usage):,} out tokens")
            wait = minutes_of_limits(batch_usage) * 60 * pace - (time.time() - b_start)
            if pending and wait > 0:
                time.sleep(wait)
    left = f" · {len(pending)} turn(s) left for the next run (time budget reached)" if pending else ""
    print(f"{len(records)}/{len(TURNS)} turns scored in {time.time() - t_start:.0f} s · run {MLFLOW_RUN_ID}{left}")

df_final = pd.DataFrame(records)
if len(df_final):
    print("Verdicts:", df_final["turn_verdict"].value_counts(dropna=False).to_dict())
    print("Answer types:", df_final["answer_type"].value_counts(dropna=False).to_dict())
    print("Evidence:", df_final["evidence_source"].value_counts(dropna=False).to_dict(),
          "· retrieval quality:", df_final["retrieval_quality"].value_counts(dropna=False).to_dict(),
          "· stage at fault:", df_final["error_source"].value_counts(dropna=False).to_dict())
    print(f"{int(df_final['n_judge_calls'].sum())} judge calls ({df_final['n_judge_calls'].mean():.1f} per turn) · "
          f"{int(df_final['judge_input_tokens'].sum()):,} in / {int(df_final['judge_output_tokens'].sum()):,} out tokens · "
          f"cost ${df_final['judge_cost_usd'].sum():.3f} · "
          f"{int((df_final['judge_errors'].map(len) > 0).sum())} turn(s) with scorer errors")
    print(f"✓ {len(records)} rows → {SCORES_TABLE} · {len(assessment_rows)} rows → {ASSESSMENTS_TABLE} · "
          f"1 row → {SCORING_RUNS_TABLE}")

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
        mlflow.log_metrics({f"error_source/{k}": int(v) for k, v in df_final["error_source"].value_counts().items()})
        by_agent = df_final.groupby("endpoint_name")["turn_verdict"].apply(lambda s: float((s == "bad").mean()))
        mlflow.log_metrics({f"bad_rate/{k}": v for k, v in by_agent.items()})
        worst = df_final[df_final["turn_verdict"] == "bad"].head(50)
        if len(worst):
            mlflow.log_table(worst[["message_id", "endpoint_name", "question_intent", "user_question", "answer",
                                    "failure_reasons", "error_source", "evidence_source", "groundedness__rationale",
                                    "missed_answer_detail", "retrieval_quality__rationale",
                                    "feedback_vote", "scoring_trace_id"]].astype(str), "worst_turns.json")
    print(f"✓ MLflow run {MLFLOW_RUN_ID}: metrics run/*, reason/*, error_source/*, bad_rate/*, artifact worst_turns.json")

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

if FAIL_ON_ALERT and alerts:
    raise RuntimeError("Quality alert: " + " | ".join(alerts))

# COMMAND ----------

# DBTITLE 1,Output schema — the dashboard reads the tables: views over the quality outputs are dropped
QUALITY_VIEW_PREFIXES = ("v_chat_quality_", "v_ka_eval_", "v_quality_shared_scorers")
if not DRY_RUN:
    _catalog, _schema = OUTPUT_SCHEMA.split(".", 1)
    for r in spark.sql(f"SELECT table_name FROM {_catalog}.information_schema.views "
                       f"WHERE table_schema = '{_schema}'").collect():
        if r["table_name"].startswith(QUALITY_VIEW_PREFIXES):
            spark.sql(f"DROP VIEW IF EXISTS {OUTPUT_SCHEMA}.{r['table_name']}")
            print(f"View dropped: {OUTPUT_SCHEMA}.{r['table_name']}")

# COMMAND ----------

# DBTITLE 1,Run summary — what was written, and where to look
if DRY_RUN:
    print("Dry run: no judge call, no table write, no MLflow run. Set dry_run=false to score.")
elif df_final.empty:
    print("No new turn to score: tables and MLflow are unchanged.")
else:
    print(f"Tables : {SCORES_TABLE} ({spark.table(SCORES_TABLE).count()} turns in total), {ASSESSMENTS_TABLE}, "
          f"{SCORING_RUNS_TABLE}")
    print(f"MLflow : {EXPERIMENT_PATH} → run {MLFLOW_RUN_ID} (Traces: one per turn; Judges: {len(SCORERS)} scorers)")
    print(f"Judge configuration: {JUDGE_CONFIG_ID}")
    display(spark.table(SCORES_TABLE).filter(F.col("scored_at") == RUN_TS)
            .select("created_at", "endpoint_name", "question_intent", "answer_type", "turn_verdict",
                    "failure_reasons", "groundedness_level", "user_question")
            .orderBy(F.col("created_at").desc()))
