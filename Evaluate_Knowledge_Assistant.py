# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Evaluation
# MAGIC
# MAGIC Calls a Knowledge Assistant endpoint on every case of the golden dataset, scores the answers with MLflow judges and
# MAGIC writes the results to Unity Catalog.
# MAGIC
# MAGIC ### What the judges read
# MAGIC - **Case**: the question (with its conversation) and the expectations of the golden dataset (expected facts,
# MAGIC   expected documents, guidelines).
# MAGIC - **Answer**: the assistant's answer, without the text fragments of citation links (`#:~:text=…`).
# MAGIC - **Evidence** (`RETRIEVER` step): the passages the assistant retrieved from the `qualibot` index, read from the
# MAGIC   trace it returns with its answer (`assistant_retrieval`). When the trace shows no retrieval step, excerpts of every
# MAGIC   document the answer relies on, searched with each line that cites it (`cited_document_excerpts`). No limit, no
# MAGIC   truncation.
# MAGIC - **Independent search** (`corpus_search`): the question searched in the whole index, to tell a retrieval miss from a
# MAGIC   documentation gap.
# MAGIC
# MAGIC ### Scorers
# MAGIC | Axis | Scorer | Type | Question answered |
# MAGIC |---|---|---|---|
# MAGIC | Correctness | `correctness` | built-in | Does the answer contain the expected facts? |
# MAGIC | | `fact_coverage` | LLM judge | full / partial / none share of the expected facts |
# MAGIC | | `fact_contradiction` | LLM judge | Does the answer contradict an expected fact? |
# MAGIC | | `refusal_handling` | LLM judge | When a refusal is expected, does the assistant refuse without inventing? |
# MAGIC | Answer | `relevance` ¹, `language_match` ¹ | LLM judge | Does the answer address the question, in the user's language? |
# MAGIC | | `expectations_guidelines` | built-in | Are the case's guidelines followed? |
# MAGIC | Faithfulness | `groundedness` ¹ | LLM judge on the evidence | Are the answer's claims supported by the evidence? |
# MAGIC | | `missed_answer` ¹ | LLM judge on the evidence | Does the answer say "not found" while the evidence holds it? |
# MAGIC | | `compliance_claim` ¹ | LLM judge on the evidence | On compliance questions, is compliance concluded only as far as the evidence establishes it? |
# MAGIC | Retrieval | `retrieval_quality` ¹ | 1-2 LLM judges | Did the assistant retrieve what the question needs (sufficient), or does an independent search of the index find more (retrieval_miss)? |
# MAGIC | | `retrieval_sufficiency` | built-in, on the evidence | Does the evidence hold what the expected answer needs? |
# MAGIC | | `document_recall` | code | Share of the expected documents returned or cited |
# MAGIC | | `reference_integrity` ¹ | code | Does every cited document code exist? |
# MAGIC | Operations | `call_ok`, `latency_s` | code | Endpoint availability and response time |
# MAGIC
# MAGIC ¹ identical in the production monitoring notebook.
# MAGIC
# MAGIC ### Outputs
# MAGIC | Where | Content |
# MAGIC |---|---|
# MAGIC | `ka_eval_runs` | one row per run: endpoint, subset, judge model, scorer configuration |
# MAGIC | `ka_eval_metrics` | one row per run × metric: score, 95% confidence interval, number of cases |
# MAGIC | `ka_eval_results` | one row per run × case: question, answer, documents, one column per metric, human rating |
# MAGIC | `ka_eval_assessments` | one row per run × case × scorer: value, numeric value, rationale, error |
# MAGIC | MLflow experiment | Runs (one per endpoint and repetition), Traces (one per case), Judges (every scorer, not scheduled), Datasets (the golden dataset, linked to every run) |
# MAGIC
# MAGIC Every row carries its run's start time, subset and scorer configuration.
# MAGIC
# MAGIC ### How to use
# MAGIC 1. Run down to **Case selection** with `run_eval=false` to check the cases and the number of calls.
# MAGIC 2. Set `run_eval=true` (start with `sample_n=5`) and read the **Report**, which writes the tables.
# MAGIC 3. Rate a few answers in **Human labels**: the agreement with the judges tells whether the scores can be trusted.

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
dbutils.widgets.text("judge_endpoint", "databricks-gpt-6-luna")     # judge model serving endpoint
dbutils.widgets.text("sample_n", "5")                               # empty = every case of the dataset
dbutils.widgets.text("case_ids", "")                                # comma-separated case ids (overrides sample_n)
dbutils.widgets.text("repeats", "1")                                # >1 measures the assistant's variability
dbutils.widgets.dropdown("run_eval", "false", ["true", "false"])
dbutils.widgets.text("output_schema", "uat_proj.qualibot")         # result tables

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

# Evidence: the passages the assistant retrieved (returned with its trace), or excerpts of the documents it cites
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"   # index queried by the assistants
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None          # source table of the index; None = read from the index definition
EXCERPTS_PER_QUERY = 3           # when the trace has no retrieval step: chunks per search of a cited document
CORPUS_SEARCH_RESULTS = 10       # chunks of the independent search of the whole index (retrieval_quality)

# Optional case metadata (question type, difficulty, expected answerability) written by the dataset builder
GOLDEN_CACHE_TABLE = "uat_landingzone.qualibot.qualibot_eval_cache"

MAX_PARALLEL_CALLS = 3           # capacity limit of the Knowledge Assistant endpoints

# Judge model rate limits (pay-per-token endpoint, shared with every other use of the model)
JUDGE_INPUT_TOKENS_PER_MINUTE = 200_000
JUDGE_OUTPUT_TOKENS_PER_MINUTE = 20_000
JUDGE_RATE_SHARE = 0.7           # share of the limits used by this notebook
JUDGE_MAX_RETRIES = 7            # retries of a call rejected for rate limit (1 s, 2 s … 60 s: about 2 minutes)
INPUT_TOKENS_PER_JUDGE_CALL = 8000    # judge prompt: instructions, case, answer and, for the evidence judges, passages
OUTPUT_TOKENS_PER_JUDGE_CALL = 400    # rationale + hidden reasoning

# Judge cost (pay-per-token, DBU per 1M tokens), from the tokens reported by the judge model
DBU_PER_M_INPUT = 1.4
DBU_PER_M_OUTPUT = 7.1
USD_PER_DBU = 0.07               # adjust to your contract price for model serving
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
os.environ["MLFLOW_GENAI_EVAL_MAX_RETRIES"] = str(JUDGE_MAX_RETRIES)      # rate-limited judge calls are retried
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

# DBTITLE 1,Traced application — assistant call with its trace, what it retrieved, independent search of the index
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


import threading

_ASSISTANT_TRACE = threading.local()     # trace returned with the last response of this worker thread


@mlflow.trace(name="knowledge_assistant", span_type="AGENT")
def call_assistant(endpoint: str, messages: list) -> dict:
    """Raw call to the Knowledge Assistant serving endpoint (Responses format with its trace, then without, then Chat
    format). The returned trace is kept aside (_ASSISTANT_TRACE) and left out of this step's output."""
    last_error, _ASSISTANT_TRACE.value = None, None
    for body in ({"input": messages, "databricks_options": {"return_trace": True}}, {"input": messages},
                 {"messages": messages}):
        try:
            raw = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        except Exception as e:
            last_error = e
            continue
        if isinstance(raw, dict) and "databricks_output" in raw:
            _ASSISTANT_TRACE.value = (raw.get("databricks_output") or {}).get("trace")
            raw = {k: v for k, v in raw.items() if k != "databricks_output"}
        return raw
    raise last_error


def assistant_trace_id(trace) -> str:
    info = (trace or {}).get("info") or {} if isinstance(trace, dict) else {}
    return str(info.get("trace_id") or info.get("request_id") or "")


def make_predict_fn(endpoint: str):
    @mlflow.trace(name="qualibot_turn", span_type="AGENT")
    def predict_fn(messages):
        question = messages[-1]["content"]
        tags = {"case_id": str(CASE_BY_QUESTION.get(question, "")), "endpoint": endpoint, "judge_model": JUDGE_MODEL or ""}
        t0 = time.time()
        try:
            raw = call_assistant(endpoint, messages)
        except Exception as e:
            mlflow.update_current_trace(tags={**tags, "call_ok": "false"})
            record_answer_context({"error": str(e)[:1000], "returned_refs": [], "cited_refs": []})
            cited_document_excerpts({})
            return ""
        raw_answer = _extract_text(raw) or ""
        answer = clean_answer(raw_answer)          # as read by the judges; the raw answer stays in knowledge_assistant
        returned = sorted(refs_from_response(raw))
        cited = sorted(code_like(raw_answer))
        latency = time.time() - t0
        # Evidence: what the assistant retrieved; excerpts of the cited documents only when its trace does not show it
        ka_trace = _ASSISTANT_TRACE.value
        steps, passages = retrieved_passages(ka_trace) if ka_trace else (0, [])
        if steps:
            source, docs = "assistant_retrieval", passages
            record_assistant_retrieval(assistant_trace_id(ka_trace), passages)
            if REFS_BY_BASE:
                corpus_search(search_query(messages))
        else:
            docs = cited_document_excerpts(evidence_queries(question, raw_answer, [(r, None) for r in returned + cited],
                                                            link_text=json.dumps(raw, ensure_ascii=False)))
            source = "cited_document_excerpts" if docs else "none"
        refs = classify_refs(cited, "\n".join(d.page_content for d in docs))
        # Short identifiers only in the tags; document lists go to the answer_context step
        mlflow.update_current_trace(tags={**tags, "call_ok": "true", "latency_s": f"{latency:.2f}"})
        record_answer_context({"returned_refs": returned, "cited_refs": cited, "error": None,
                               "evidence_source": source, "assistant_retrieval_steps": steps,
                               "assistant_trace_returned": ka_trace is not None,
                               "evidence_refs": sorted({d.metadata["doc_uri"] for d in docs}), "evidence_count": len(docs),
                               **{f"{k}_refs": v for k, v in refs.items()}})
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
    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    found = {key(r) for r in (context.get("returned_refs") or []) + (context.get("cited_refs") or [])}
    hit = [r for r in expected if key(r) in found]
    return Feedback(value=round(len(hit) / len(expected), 3), rationale=f"expected: {expected} · found: {hit}")


@scorer(name="operations", description="call_ok: the endpoint answered · latency_s: response time of the assistant.")
def operations(outputs, trace):
    """call_ok: the endpoint answered · latency_s: response time of the assistant."""
    from mlflow.entities import Feedback

    tags = trace.info.tags or {}
    context = next((s.outputs for s in trace.search_spans(name="answer_context") if isinstance(s.outputs, dict)), {})
    ok = tags.get("call_ok") == "true" or (tags.get("call_ok") is None and bool(str(outputs or "").strip()))
    feedbacks = [Feedback(name="call_ok", value=ok, rationale=context.get("error") or None)]
    if tags.get("latency_s"):
        feedbacks.append(Feedback(name="latency_s", value=float(tags["latency_s"])))
    return feedbacks


def build_scorers(model):
    """(LLM judges called for every case, trace judges called when applicable, code scorers)."""
    kw = {"model": model} if model else {}
    every_case = [Correctness(**kw), ExpectationsGuidelines(**kw)] + \
                 [make_judge(name=n, description=d, feedback_value_type=t, instructions=i, **kw)
                  for n, (d, t, i) in EVALUATION_JUDGES.items()] + shared_llm_judges(model)
    return (every_case, [groundedness, missed_answer, retrieval_quality, compliance_claim, retrieval_sufficiency],
            [reference_integrity, document_recall, operations])

# COMMAND ----------

# DBTITLE 1,Judge check and registration — the evaluation stops if the judge model does not answer
import inspect

JUDGE_MODEL = f"databricks:/{JUDGE_ENDPOINT}" if JUDGE_ENDPOINT else None
_sample = {"inputs": {"messages": [{"role": "user", "content": "What is the retention period of inspection records?"}]},
           "outputs": "According to QP-1457, inspection records are kept for 10 years.",
           "expectations": {"expected_facts": ["Inspection records are kept for 10 years."]}}


_every_case = build_scorers(JUDGE_MODEL)[0]
for _judge in (_every_case[0], next(j for j in _every_case if j.name == "fact_coverage")):   # built-in and custom judge
    try:
        _error = getattr(_judge(**_sample), "error", None)
    except Exception as e:
        _error = e
    if _error:
        raise RuntimeError(f"Judge model {JUDGE_MODEL or 'Databricks-managed'} does not answer (check the endpoint name "
                           f"in Serving): {str(_error)[:300]}")
LLM_JUDGES, TRACE_JUDGES, CODE_SCORERS = build_scorers(JUDGE_MODEL)
SCORERS = LLM_JUDGES + TRACE_JUDGES + CODE_SCORERS
SCORERS_CONFIG_ID = scorers_config_id(SCORERS, JUDGE_MODEL, inspect.getsource(retrieved_passages),
                                      inspect.getsource(evidence_queries), EXCERPTS_PER_QUERY, CORPUS_SEARCH_RESULTS)

# Judge pacing: MLflow limits the scorer calls per second. Code scorers and skipped judges take a slot too, hence the
# ratio of scorers to judge calls.
_judge_calls_per_minute = JUDGE_RATE_SHARE * min(JUDGE_INPUT_TOKENS_PER_MINUTE / INPUT_TOKENS_PER_JUDGE_CALL,
                                                 JUDGE_OUTPUT_TOKENS_PER_MINUTE / OUTPUT_TOKENS_PER_JUDGE_CALL)
SCORER_CALLS_PER_SECOND = _judge_calls_per_minute / 60 * len(SCORERS) / (len(LLM_JUDGES) + len(TRACE_JUDGES))
os.environ["MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT"] = f"{SCORER_CALLS_PER_SECOND:.3f}"
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
_judge_calls = n * len(ENDPOINTS) * REPEATS * (len(LLM_JUDGES) + len(TRACE_JUDGES))
print(f"Estimated calls: {n * len(ENDPOINTS) * REPEATS} assistant calls, ~{_judge_calls} judge calls at most, paced "
      f"at {_judge_calls_per_minute:.0f} judge calls per minute (at least {_judge_calls / _judge_calls_per_minute:.0f} min)")
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
                mlflow.log_params({"n_cases": n, "repeat": rep, "excerpts_per_query": EXCERPTS_PER_QUERY,
                                   "judge_rate_share": JUDGE_RATE_SHARE})
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
    "groundedness": "answer supported by its evidence (partial = 0.5)",
    "missed_answer": "no information missed that the evidence contains",
    "retrieval_quality": "the assistant retrieved what the question needs (retrieval_miss = 0; documentation gaps excluded)",
    "compliance_claim": "compliance concluded only as far as the evidence establishes it (compliance questions)",
    "retrieval_sufficiency": "evidence contains what the expected answer needs",
    "document_recall": "expected documents returned or cited",
    "reference_integrity": "every cited document code exists",
    "expectations_guidelines": "case-specific guidelines followed",
    "call_ok": "endpoint answered",
    "latency_s": "response time in seconds (median in ka_eval_metrics)",
}
FAILURE_METRICS = ["correctness", "fact_contradiction", "refusal_handling", "relevance", "groundedness", "missed_answer",
                   "retrieval_quality", "compliance_claim", "reference_integrity", "expectations_guidelines", "call_ok"]

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
    ("assistant_retrieval_coverage", D, "Share of cases whose evidence is the assistant's own retrieval (from its trace)"),
    ("judge_input_tokens", I, "Input tokens of the judge calls, as reported by the judge model"),
    ("judge_output_tokens", I, "Output tokens of the judge calls, as reported by the judge model"),
    ("judge_cost_usd", D, "Judge cost (USD), from the tokens"),
]
METRICS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("started_at", S, "Start time (UTC, ISO 8601) of the run"),
    ("subset", S, "full, sample:<n>:seed<seed> or cases:<ids>"),
    ("scorers_config_id", S, "Fingerprint of the scorer definitions and judge model: compare runs of the same configuration"),
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
    ("started_at", S, "Start time (UTC, ISO 8601) of the run"),
    ("subset", S, "full, sample:<n>:seed<seed> or cases:<ids>"),
    ("scorers_config_id", S, "Fingerprint of the scorer definitions and judge model: compare runs of the same configuration"),
    ("case_id", S, "Golden case id (question id of the dataset builder)"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("question", S, "Question of the case (last user message)"),
    ("intent", S, "Question type assigned by the dataset builder"),
    ("difficulty", S, "easy, medium or hard (dataset builder)"),
    ("final_answerability", S, "full, partial, none or out_of_scope: whether the documentation answers the question"),
    ("expected", S, "Expected facts (one per line), or the expected answer for refusal cases"),
    ("expected_sources", A, "Documents the answer should rely on"),
    ("answer", S, "Assistant answer, without citation text fragments"),
    ("returned_refs", A, "Documents returned by the assistant as citations"),
    ("cited_refs", A, "Document codes cited in the answer text"),
    ("unverified_refs", A, "Cited codes found neither in the index nor in the evidence"),
    ("evidence_source", S, "Evidence read by the judges: assistant_retrieval (passages the assistant retrieved, from its "
                           "trace) or cited_document_excerpts (when its trace shows no retrieval)"),
    ("evidence_count", I, "Evidence passages read by the judges"),
    ("trace_id", S, "MLflow trace of the case"),
    *[(m, D, f"Numeric score: {d}") for m, d in METRICS.items()],
    ("human_fact_coverage", D, "Human rating of the expected facts present (full=1, partial=0.5, none=0)"),
    ("failed_scorers", A, "Scorers that failed on this case (score 0)"),
]
ASSESSMENTS_COLUMNS = [
    ("run_id", S, "MLflow run id of the evaluation run"),
    ("started_at", S, "Start time (UTC, ISO 8601) of the run"),
    ("subset", S, "full, sample:<n>:seed<seed> or cases:<ids>"),
    ("scorers_config_id", S, "Fingerprint of the scorer definitions and judge model: compare runs of the same configuration"),
    ("case_id", S, "Golden case id"),
    ("endpoint", S, "Knowledge Assistant serving endpoint evaluated"),
    ("trace_id", S, "MLflow trace of the case"),
    ("assessment_name", S, "Scorer name, or the name of a human rating"),
    ("source_type", S, "LLM_JUDGE, CODE or HUMAN"),
    ("value", S, "Value as returned by the scorer"),
    ("value_numeric", D, "Numeric form: 1 = pass, 0 = fail, 0.5 = partial; counts and durations as is; NULL for labels"),
    ("rationale", S, "Rationale of the scorer"),
    ("error", S, "Error message when the scorer failed"),
    ("judge_input_tokens", I, "Input tokens of the judge call(s), as reported by the judge model"),
    ("judge_output_tokens", I, "Output tokens of the judge call(s), as reported by the judge model"),
]
TABLES = {
    RUNS_TABLE: (RUNS_COLUMNS, "Qualibot Knowledge Assistant evaluation runs on the golden dataset: one row per run."),
    METRICS_TABLE: (METRICS_COLUMNS, "Qualibot evaluation metrics: one row per run and metric, with 95% confidence intervals."),
    RESULTS_TABLE: (RESULTS_COLUMNS, "Qualibot evaluation results: one row per run and golden case, one column per metric."),
    ASSESSMENTS_TABLE: (ASSESSMENTS_COLUMNS, "Qualibot evaluation assessments: one row per run, golden case and scorer, "
                                             "human ratings included."),
}
SCHEMAS = {t: StructType([StructField(n, dt) for n, dt, _ in cols]) for t, (cols, _) in TABLES.items()}
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
        out.append({"trace_id": t.info.trace_id, "tags": t.info.tags or {}, "context": answer_context(t), "answer": answer,
                    "assessments": [assessment_row(a) for a in (t.info.assessments or [])
                                    if type(a).__name__ != "Expectation" and getattr(a, "expectation", None) is None]})
    return out


def collect(run_id: str) -> pd.DataFrame:
    """One row per case: numeric scores (human ratings prefixed human::), rationales (_why), answer, case metadata."""
    rows = []
    for tr in run_traces(run_id):
        row, why = {"trace_id": tr["trace_id"], "case_id": tr["tags"].get("case_id"), "answer": tr["answer"],
                    "_tags": tr["tags"], "_context": tr["context"], "_assessments": tr["assessments"]}, {}
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
    """Writes (or rewrites) one evaluation run in the four tables."""
    run = mlflow.get_run(run_id)
    tags, params = run.data.tags, run.data.params
    endpoint = tags.get("endpoint")
    run_row = {"run_id": run_id, "run_name": run.info.run_name, "endpoint": endpoint,
               "started_at": pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").isoformat(),
               "dataset_name": tags.get("dataset"), "subset": tags.get("subset"), "n_cases": params.get("n_cases"),
               "repeat": params.get("repeat"), "judge_model": tags.get("judge_model"),
               "scorers_config_id": tags.get("scorers_config_id"), "experiment_id": EXPERIMENT_ID}
    judged = [a for _, r in res.iterrows() for a in r["_assessments"] if a["source_type"] == "LLM_JUDGE"]
    tokens_in = sum(a.get("input_tokens") or 0 for a in judged)
    tokens_out = sum(a.get("output_tokens") or 0 for a in judged)
    run_row.update({"assistant_retrieval_coverage": float((res["_context"].map(
                        lambda c: c.get("evidence_source") == "assistant_retrieval")).mean()) if len(res) else None,
                    "judge_input_tokens": tokens_in, "judge_output_tokens": tokens_out,
                    "judge_cost_usd": round((tokens_in * DBU_PER_M_INPUT + tokens_out * DBU_PER_M_OUTPUT) / 1e6
                                            * USD_PER_DBU, 6)})
    context = {k: run_row[k] for k in ("started_at", "subset", "scorers_config_id")}
    metric_rows = [{"run_id": run_id, "endpoint": endpoint, **context, **r} for r in summary.to_dict("records")]
    result_rows, assessment_rows = [], []
    for _, r in res.iterrows():
        exp = r.get("expectations") if isinstance(r.get("expectations"), dict) else {}
        ctx = r["_context"]
        result_rows.append({
            "run_id": run_id, **context, "case_id": r["case_id"], "endpoint": endpoint, "question": r.get("question"),
            "intent": r.get("intent"), "difficulty": r.get("difficulty"),
            "final_answerability": r.get("final_answerability"), "expected": _expected_text(exp),
            "expected_sources": [clean_ref(d["doc_uri"]) for d in exp.get("expected_retrieved_context", [])],
            "answer": str(r.get("answer") or ""),
            "returned_refs": ctx.get("returned_refs") or [], "cited_refs": ctx.get("cited_refs") or [],
            "unverified_refs": ctx.get("unverified_refs") or [], "evidence_source": ctx.get("evidence_source"),
            "evidence_count": ctx.get("evidence_count"), "trace_id": r["trace_id"],
            **{m: r.get(m) for m in METRICS}, "human_fact_coverage": r.get("human::fact_coverage"),
            "failed_scorers": [m for m in FAILURE_METRICS if r.get(m) == 0]})
        assessment_rows += [{"run_id": run_id, **context, "case_id": r["case_id"], "endpoint": endpoint, "trace_id": r["trace_id"],
                             "assessment_name": a["name"], "source_type": a["source_type"],
                             "value": a["value"] if isinstance(a["value"], str) or a["value"] is None else json.dumps(a["value"]),
                             "value_numeric": numeric_value(a["name"], a["value"]), "rationale": a["rationale"],
                             "error": a["error"], "judge_input_tokens": a.get("input_tokens"),
                             "judge_output_tokens": a.get("output_tokens")} for a in r["_assessments"]]
    for table, rows in [(RUNS_TABLE, [run_row]), (METRICS_TABLE, metric_rows), (RESULTS_TABLE, result_rows),
                        (ASSESSMENTS_TABLE, assessment_rows)]:
        cols, comment = TABLES[table]
        ensure_table(table, SCHEMAS[table], comment, {n: d for n, _, d in cols})
        replace_rows(table, SCHEMAS[table], rows, ["run_id"])
    print(f"✓ run {run_id} written to {RUNS_TABLE}, {METRICS_TABLE}, {RESULTS_TABLE} ({len(result_rows)} cases), "
          f"{ASSESSMENTS_TABLE} ({len(assessment_rows)} rows) · evidence from the assistant's retrieval: "
          f"{run_row['assistant_retrieval_coverage'] or 0:.0%} of the cases · judge cost ${run_row['judge_cost_usd']:.3f}")

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
            mlflow.log_table(failures.drop(columns=["_why", "_tags", "_context", "_assessments", "expectations"],
                                           errors="ignore")
                             .astype(str), "report/failures.json")
    print("✓ report logged to the run (Metrics: report/*, Artifacts: report/)")
    write_run_tables(run_id, res, summary)

# COMMAND ----------

# DBTITLE 1,Run comparison — from ka_eval_metrics; compare runs of the same subset and scorer configuration
if spark.catalog.tableExists(METRICS_TABLE):
    history = spark.table(METRICS_TABLE).select("started_at", "endpoint", "subset", "scorers_config_id", "metric",
                                                "score").toPandas()
    table = (history.pivot_table(index=["started_at", "endpoint", "subset", "scorers_config_id"], columns="metric",
                                 values="score").sort_index(ascending=False).head(20).round(3))
    display(table.reset_index())
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
                f"**Documents returned by the assistant**: {', '.join(r['_context'].get('returned_refs') or []) or '—'}",
                f"**Documents cited in the answer**: {', '.join(r['_context'].get('cited_refs') or []) or '—'}",
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
