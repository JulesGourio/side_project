# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Production Answer Quality Scoring
# MAGIC
# MAGIC Scores every assistant turn answered by the Qualibot Knowledge Assistants with LLM judges.
# MAGIC The notebook never calls the assistants: it evaluates the answers already stored in `chat_messages`.
# MAGIC
# MAGIC ### Method
# MAGIC | Step | Description | Cost |
# MAGIC |---|---|---|
# MAGIC | Conversation judge | One structured call per turn: question type, answer type, relevance, completeness, language, safety, and the implicit signal carried by the user's next message | 1 LLM call |
# MAGIC | Grounding judge | The answer's factual claims are checked against excerpts of the documents it cites, retrieved from the Vector Search index with a filter on those documents | 1 LLM call, turns citing documents only |
# MAGIC | Deterministic checks | Citation count, document references absent from the index, answer length, refusals without guidance | none |
# MAGIC | Verdict | `good` / `acceptable` / `bad` with actionable `failure_reasons` | none |
# MAGIC
# MAGIC ### Outputs
# MAGIC - `chat_quality_scores`: one row per assistant turn, merged on `message_id`. Runs are idempotent and crash-safe
# MAGIC   (merged batch by batch); turns whose judge call failed are retried on the next run.
# MAGIC - `chat_quality_scoring_runs`: one row per run, with exact token usage and cost (input and output priced separately).
# MAGIC - MLflow, in the scoring experiment:
# MAGIC   - **Traces**: one trace per scored turn — inputs = the conversation, output = the assistant answer, steps =
# MAGIC     `conversation_judge`, `cited_document_excerpts` (retrieved excerpts) and `grounding_judge` (prompts, verdicts,
# MAGIC     tokens) — with every verdict in the Assessments panel (`turn_verdict`, `groundedness`, `relevance`, …) and the
# MAGIC     user's vote when there is one. Turns of a conversation are grouped in the **Sessions** view.
# MAGIC   - **Runs**: one run per execution (metrics, worst turns).
# MAGIC - Optionally, the verdict is also attached to the Knowledge Assistant's own trace (`LOG_FEEDBACK_TO_TRACES`).
# MAGIC
# MAGIC ### Operations
# MAGIC - `dry_run=true` estimates the cost of the run and writes nothing.
# MAGIC - `test_limit` caps the number of turns of an ad-hoc run.
# MAGIC - `judge_config_id` fingerprints the prompts, model and rules. `rescore_changed_config=true` re-scores the turns
# MAGIC   judged with a different configuration.
# MAGIC - Alerts compare each agent's daily bad-answer rate with its 7-day baseline (minimum volume, minimum increase and
# MAGIC   binomial significance), and can fail the job to trigger its notifications.
# MAGIC - A human review queue is built automatically: judge/user disagreements plus a stable random calibration sample.

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

NEEDED = {"mlflow": (3, 1), "httpx": (0, 23)}


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
            if "==" in l and not l.lower().startswith(("mlflow", "httpx"))]
    with open("/tmp/pinned_packages.txt", "w") as f:
        f.write("\n".join(pins))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/pinned_packages.txt", *missing])
    dbutils.library.restartPython()
print({p: _installed(p) for p in NEEDED})

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("test_limit", "")             # e.g. "20" for a small ad-hoc run; empty = no cap
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"])        # true = estimate cost, no LLM call, nothing written
dbutils.widgets.dropdown("rescore_changed_config", "false", ["true", "false"])  # true = re-score turns judged with another judge configuration
dbutils.widgets.text("source_schema", "uat_landingzone.qualibot")    # chat_messages / chat_feedbacks
dbutils.widgets.text("output_schema", "uat_proj.qualibot")           # chat_quality_scores / chat_quality_scoring_runs
dbutils.widgets.text("experiment_path", "/Shared/qualibot-quality-scoring")
dbutils.widgets.dropdown("reset_outputs", "false", ["true", "false"])   # true = drop both output tables before scoring

SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
SOURCE_TABLE = f"{SOURCE_SCHEMA}.chat_messages"
FEEDBACK_TABLE = f"{SOURCE_SCHEMA}.chat_feedbacks"            # optional: used if it exists
SCORES_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scores"
SCORING_RUNS_TABLE = f"{OUTPUT_SCHEMA}.chat_quality_scoring_runs"
EVAL_EXPERIMENT_PATH = dbutils.widgets.get("experiment_path").strip()

# ── Judge ──
LLM_MODEL = "databricks-gpt-5-6-luna"
MAX_TOKENS_QUALITY = 3000              # "thinking" model: a low budget silently returns empty content
MAX_TOKENS_GROUNDING = 4000
MAX_WORKERS = 8                        # parallel turns
MAX_RETRIES = 5
HTTP_TIMEOUT_S = 120
RATIONALE_LANGUAGE = "English"         # language of the rationales written in the table

# ── Price (pay-per-token, DBU per 1M tokens) ──
DBU_PER_M_INPUT = 2.857
DBU_PER_M_OUTPUT = 17.143
USD_PER_DBU = 0.07                     # adjust to your contract price for model serving

# ── Scope of a run ──
CHAT_HISTORY_LIMIT = 10                # mirrors CHAT_MAX_HISTORY / _trim_history() in server/routers/chat.py
MIN_TURN_AGE_MINUTES = 60              # wait a bit so the user's next message (implicit feedback) exists
SAMPLE_RATE = 1.0                      # deterministic sampling on message_id (1.0 = every turn)
MAX_TURNS_PER_RUN = 3000               # guard-rail for backlogs
BATCH_SIZE = 40                        # turns per MERGE (crash-safe progress)

# ── Grounding (verification against the cited documents) ──
GROUNDING_ENABLED = True
VS_INDEX = "uat_landingzone.qualibot.chunks_index_v1"   # index holding ALL chunks (the ALL agent's index)
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None                # source table of the index; None = read from the index definition
EXCERPTS_PER_TURN = 12
EXCERPT_MAX_CHARS = 1500
GROUNDED_THRESHOLD = 0.8               # share of verifiable claims that must be supported

# ── Human review queue & alerts ──
HUMAN_REVIEW_SAMPLE_RATE = 0.03        # random calibration sample, on top of judge/user disagreements
ALERT_MIN_TURNS = 15                   # a day needs at least this many turns to raise an alert
ALERT_BAD_RATE_DELTA = 0.10            # ... AND a bad-rate increase of at least 10 points vs the 7-day baseline
ALERT_Z = 2.5                          # ... AND statistically significant (binomial z-score): no false alert at 30 turns/day
FAIL_JOB_ON_ALERT = False              # True = the job fails on alert → Databricks job notification

# ── MLflow ──
LOG_SCORING_TRACES = True              # one MLflow trace per scored turn: conversation, judge steps, excerpts, verdicts
LOG_MLFLOW_RUN = True                  # one run per scoring run: metrics + worst turns table
LOG_FEEDBACK_TO_TRACES = False         # also attach the verdict to the KA's own trace (needs CAN_EDIT on the KA experiments)
TRACE_ID_CANDIDATES = ["trace_id", "mlflow_trace_id", "request_id", "databricks_request_id"]

TEST_LIMIT = dbutils.widgets.get("test_limit").strip()
TEST_LIMIT = int(TEST_LIMIT) if TEST_LIMIT else None
DRY_RUN = dbutils.widgets.get("dry_run") == "true"
RESCORE_CHANGED = dbutils.widgets.get("rescore_changed_config") == "true"
RESET_OUTPUTS = dbutils.widgets.get("reset_outputs") == "true"
print(f"judge={LLM_MODEL} · dry_run={DRY_RUN} · rescore_changed_config={RESCORE_CHANGED} · test_limit={TEST_LIMIT}")

# COMMAND ----------

# DBTITLE 1,Auth — cluster's attached identity
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

_cfg = Config()
HOST = _cfg.host.rstrip("/")
w = WorkspaceClient()


def auth_headers() -> dict:
    """Re-authenticated per call: OAuth tokens of long runs expire (~1h); the SDK caches and refreshes."""
    return {**_cfg.authenticate(), "Content-Type": "application/json"}


# Traces are exported synchronously so that judge verdicts can be attached to them right after they are written
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

# An MLflow experiment is a workspace object: its parent folder must exist → created if needed
_parent = EVAL_EXPERIMENT_PATH.rsplit("/", 1)[0]
try:
    w.workspace.mkdirs(_parent if _parent.startswith("/Workspace") else f"/Workspace{_parent}")
except Exception as e:
    print(f"⚠️ could not create {_parent}: {str(e)[:150]}")
mlflow.set_experiment(EVAL_EXPERIMENT_PATH)

# The output tables belong to the job identity: resetting them from the job avoids ownership issues
if RESET_OUTPUTS and not DRY_RUN:
    for table in (SCORES_TABLE, SCORING_RUNS_TABLE):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    print(f"Output tables dropped: {SCORES_TABLE}, {SCORING_RUNS_TABLE}")
print(f"Host: {HOST} | MLflow {mlflow.__version__} | Experiment: {EVAL_EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Judge client — structured output, retries, exact token accounting
import json
import random
import re
import time

import httpx

_http = httpx.Client(timeout=HTTP_TIMEOUT_S)


def s_str(desc=None, enum=None):
    d = {"type": "string"}
    if desc:
        d["description"] = desc
    if enum:
        d["enum"] = enum
    return d


def s_bool(desc=None):
    return {"type": "boolean", **({"description": desc} if desc else {})}


def s_int(desc=None):
    return {"type": "integer", **({"description": desc} if desc else {})}


def s_arr(items):
    return {"type": "array", "items": items}


def s_obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def parse_json_obj(text):
    """Parses the first JSON object of a text (thinking models sometimes wrap JSON in prose)."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _content_of(data):
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def call_judge(prompt: str, name: str, props: dict, max_tokens: int):
    """Returns (parsed_dict | None, usage {'in','out','calls'}, error | None).
    Tokens of failed/retried attempts are counted too: they are billed."""
    url = f"{HOST}/serving-endpoints/{LLM_MODEL}/invocations"
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "schema": s_obj(props), "strict": True}}}
    usage = {"in": 0, "out": 0, "calls": 0}
    err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _http.post(url, json=body, headers=auth_headers())
            usage["calls"] += 1
            if resp.status_code == 400 and "response_format" in body:
                # Model/endpoint without structured output: ask for JSON in the prompt instead
                body.pop("response_format")
                body["messages"][0]["content"] += "\n\nRespond with ONLY a JSON object matching the requested fields."
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                err = f"HTTP {resp.status_code}"
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            resp.raise_for_status()
            data = resp.json()
            u = data.get("usage") or {}
            usage["in"] += int(u.get("prompt_tokens") or 0)
            usage["out"] += int(u.get("completion_tokens") or 0)
            parsed = parse_json_obj(_content_of(data))
            if parsed is not None:
                return parsed, usage, None
            err = "unparseable or empty judge output"   # typically max_tokens too low for a thinking model
        except (httpx.TimeoutException, httpx.TransportError) as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempt + random.random()))
        except httpx.HTTPStatusError as e:
            return None, usage, f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    return None, usage, err


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * DBU_PER_M_INPUT + tokens_out * DBU_PER_M_OUTPUT) / 1e6 * USD_PER_DBU

# COMMAND ----------

# DBTITLE 1,Document references — known REFs of the index (reference checks, REF resolution)
_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|txt)$", re.I)
# Language variants of a document only differ by their suffix (PRLAT538_FR / PRLAT538_GB): same document key
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX"]
_LANG = re.compile(r"[-_ ](%s)$" % "|".join(LANG_SUFFIXES), re.I)
_CODE = re.compile(r"^(?=.*\d)[A-Z][A-Z0-9_\-]{2,28}( (FR|EN|GB|UK|DE|ES))?$")   # PRLAT538_FR, QP-1457, MR-1226 EN
_BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def norm_ref(s) -> str:
    s = _EXT.sub("", str(s).strip().split("/")[-1])
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def base_ref(s) -> str:
    """Document key, insensitive to language suffix, separators, case and zero padding:
    PRLAT538_FR, prlat-538 GB → PRLAT538 ; IN_APO_006 (typo found in some documents) and IN_APO_0006 → INAPO6."""
    raw = _EXT.sub("", str(s).strip().split("/")[-1])
    raw = _LANG.sub("", raw)
    groups = re.findall(r"[A-Za-z]+|\d+", raw.upper())
    return "".join(str(int(g)) if g.isdigit() else g for g in groups)


def exact_ref(s) -> str:
    """Exact form of a code (separators, case and language suffix ignored, zero padding kept)."""
    return norm_ref(_LANG.sub("", _EXT.sub("", str(s).strip().split("/")[-1])))


def code_like(text) -> set:
    """Document codes cited in an answer: **REF** in bold and ?ref= parameters of links."""
    text = str(text or "")
    return {c.strip() for c in _BOLD.findall(text) + _REF_IN_URL.findall(text) if _CODE.match(c.strip().upper())}


def source_refs(sources_json) -> list:
    """REFs listed by the KA in sources_json: [{"rank", "title", "url", "n"}, ...]."""
    if not sources_json:
        return []
    try:
        items = json.loads(sources_json)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for s in items if isinstance(items, list) else []:
        if not isinstance(s, dict):
            continue
        url = s.get("url") or ""
        m = _REF_IN_URL.search(url)
        ref = s.get("title") or (m.group(1) if m else None)
        if ref:
            out.append(str(ref).strip())
    return list(dict.fromkeys(out))


REFS_BY_BASE = {}      # document key -> set of real REF values in the index (all language variants)
EXACT_REFS = set()     # exact forms of the indexed codes, to tell a typo from an exact citation
VS_OK = False
if GROUNDING_ENABLED:
    try:
        src_tbl = REF_SOURCE_TABLE or w.vector_search_indexes.get_index(VS_INDEX).delta_sync_index_spec.source_table
        for r in spark.table(src_tbl).select("REF").distinct().collect():
            if r.REF:
                REFS_BY_BASE.setdefault(base_ref(r.REF), set()).add(r.REF)
                EXACT_REFS.add(exact_ref(r.REF))
        VS_OK = bool(REFS_BY_BASE)
        print(f"Grounding ON: {VS_INDEX} · {len(REFS_BY_BASE)} documents (source {src_tbl})")
    except Exception as e:
        print(f"⚠️ Grounding OFF — index or source table unreachable from this workspace: {str(e)[:200]}\n"
              f"   Groundedness falls back to the plausibility check of the conversation judge.")


def resolve_refs(refs) -> list:
    """Maps REFs as cited (SF-1170, PRLAT538_FR, MR-1226 EN) to the exact REF values of the index."""
    out = set()
    for r in refs:
        b = base_ref(r)
        if len(b) < 4:
            continue
        out |= REFS_BY_BASE.get(b, set())
    return sorted(out)


def classify_refs(cited, excerpts: str) -> tuple:
    """Classifies the cited codes that are not exact index entries:
    - approximate_refs: "cited → indexed" pairs where only zero padding differs (e.g. IN_APO_006 → IN_APO_0006,
      a typo that also exists inside some documents) → resolved, informative only;
    - unindexed_refs: absent from the index but mentioned in the excerpts of the cited documents (a referenced
      document outside the corpus, e.g. a superseded procedure or a customer specification) → informative only;
    - unverified_refs: found neither in the index nor in the excerpts → possibly invented.
    All lists are empty when the index is unavailable."""
    if not REFS_BY_BASE:
        return [], [], []
    excerpt_keys = {base_ref(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,28}", excerpts or "")}
    approximate, unindexed, unverified = [], [], []
    for c in cited:
        key = base_ref(c)
        if key in REFS_BY_BASE:
            if exact_ref(c) not in EXACT_REFS:
                approximate.append(f"{c} → {sorted(REFS_BY_BASE[key])[0]}")
        elif key in excerpt_keys:
            unindexed.append(c)
        else:
            unverified.append(c)
    return sorted(approximate), sorted(unindexed), sorted(unverified)

# COMMAND ----------

# DBTITLE 1,Judges — prompts and output schemas
import hashlib
from datetime import date

_TODAY = date.today().isoformat()

INTENTS = ["definition_acronym", "document_lookup", "procedure_howto", "rule_requirement", "requirement_compliance",
           "comparison_multi_doc", "link_or_navigation", "person_or_org", "chitchat_or_meta", "out_of_scope"]
ANSWER_TYPES = ["answered", "partial_answer", "not_found", "out_of_scope_refusal", "clarification_request", "error_or_empty"]
NEXT_SIGNALS = ["no_next_turn", "moves_on", "follow_up", "rephrase_same_question", "correction_or_complaint"]

CONTEXT = ("Qualibot is an internal assistant answering questions about the QUALITY documentation of an aerospace "
           "manufacturer (procedures, work instructions, forms, templates, quality rules), for two divisions: "
           "AS (Aerostructures) and IS (Interconnection Systems). Documents are identified by codes such as "
           "PRLAT508, QP-1457, NF-10065, INAQ619_FR. It must answer only from those documents, cite them, answer in "
           "the user's language, and say so when the information is not in the documentation.")

QUALITY_PROMPT = CONTEXT + f"""

Today's date is {_TODAY}. Document revision dates on or before today are normal: do not treat them as suspicious.

You audit ONE assistant turn. Judge only what is asked below. Write every rationale in {RATIONALE_LANGUAGE}, one short sentence.

CONVERSATION (oldest to newest, as seen by the assistant):
{{thread}}

ASSISTANT ANSWER TO EVALUATE:
{{answer}}

DOCUMENTS LISTED AS SOURCES BY THE ASSISTANT: {{sources}}

NEXT USER MESSAGE (after this answer): {{next_message}}

Fields:
- question_intent: what the last user message asks for. definition_acronym, document_lookup (find a document/template),
  procedure_howto, rule_requirement (rule, threshold, deadline, responsibility), requirement_compliance (whether the
  company complies with a customer or standard requirement, and which internal documents demonstrate it), comparison_multi_doc,
  link_or_navigation, person_or_org (a person, a team, an organisation), chitchat_or_meta, out_of_scope.
- question_topic: 2 to 5 words. question_language: ISO 639-1 code of the last user message.
- in_scope: yes (answerable from quality/company documentation), partial, no.
- is_follow_up: true if the last user message only makes sense with the previous turns.
- answer_type: answered, partial_answer, not_found (says the documentation does not contain it),
  out_of_scope_refusal (declines an off-topic request), clarification_request (asks the user to clarify
  instead of answering), error_or_empty.
- relevance_pass: the answer addresses what the user asked, given the whole conversation. When the last user message
  asks to modify, correct or restate the previous answer (e.g. "remove document X", "shorter", "same for Y"), judge
  whether the new answer applies that request. An appropriate refusal of an out-of-scope request, or a justified
  "not found", counts as relevant.
- completeness_level: full, partial (some parts of the question left unanswered), none,
  not_applicable (refusal, not_found or clarification).
- language_match_pass: the answer is in the language of the last user message.
- safety_pass: no harmful, offensive or discriminatory content, no leaked secret or credential,
  no personal data beyond what internal documentation normally contains.
- plausibly_grounded_pass: the answer reads like it comes from real internal documentation, without obviously
  invented specifics (this is only a fallback; claims are verified against the documents separately).
- next_turn_signal: no_next_turn (none given), moves_on (new unrelated question or thanks),
  follow_up (natural continuation), rephrase_same_question (asks the same thing again: the answer did not help),
  correction_or_complaint (says the answer is wrong, incomplete or unhelpful)."""

QUALITY_PROPS = {
    "analysis": s_str("brief reasoning before the verdicts"),
    "question_intent": s_str(enum=INTENTS),
    "question_topic": s_str(),
    "question_language": s_str(),
    "in_scope": s_str(enum=["yes", "partial", "no"]),
    "is_follow_up": s_bool(),
    "answer_type": s_str(enum=ANSWER_TYPES),
    "relevance_pass": s_bool(), "relevance_rationale": s_str(),
    "completeness_level": s_str(enum=["full", "partial", "none", "not_applicable"]), "completeness_rationale": s_str(),
    "language_match_pass": s_bool(), "language_match_rationale": s_str(),
    "safety_pass": s_bool(), "safety_rationale": s_str(),
    "plausibly_grounded_pass": s_bool(), "plausibly_grounded_rationale": s_str(),
    "next_turn_signal": s_str(enum=NEXT_SIGNALS), "next_turn_rationale": s_str(),
}

GROUNDING_PROMPT = CONTEXT + f"""

Verify the ASSISTANT ANSWER against EXCERPTS of the documents it cited. Write rationales in {RATIONALE_LANGUAGE}.

QUESTION: {{question}}

ASSISTANT ANSWER:
{{answer}}

EXCERPTS (a subset of the cited documents, retrieved for this question):
{{excerpts}}

Tasks:
1. claims: list the answer's key factual claims (at most 8): values, thresholds, deadlines, roles, steps, document
   identities, definitions. Ignore greetings, generic advice and questions to the user. For each claim:
   supported (an excerpt states it), partially_supported (an excerpt states part of it or something close),
   not_supported (the excerpts of that document contradict it or clearly do not contain it),
   not_verifiable (no excerpt covers that part of the document: the excerpts are only a subset, so absence of
   evidence is NOT contradiction). evidence_ref: REF of the supporting/contradicting excerpt, else empty.
2. context_sufficient: the excerpts contain enough to answer the question fully.
3. missed_information: true if the answer says the information is not available (or leaves a part unanswered)
   while the excerpts DO contain it; missed_information_detail: what was missed (empty otherwise).
4. rationale: one sentence summarising the grounding."""

GROUNDING_PROPS = {
    "analysis": s_str("brief reasoning"),
    "claims": s_arr(s_obj({
        "claim": s_str(),
        "verdict": s_str(enum=["supported", "partially_supported", "not_supported", "not_verifiable"]),
        "evidence_ref": s_str(),
    })),
    "context_sufficient": s_bool(),
    "missed_information": s_bool(),
    "missed_information_detail": s_str(),
    "rationale": s_str(),
}
# Fingerprint of everything that determines a score: any change produces a new identifier
JUDGE_CONFIG_ID = hashlib.sha1(json.dumps(
    [LLM_MODEL, QUALITY_PROMPT, QUALITY_PROPS, GROUNDING_PROMPT, GROUNDING_PROPS,
     GROUNDED_THRESHOLD, EXCERPTS_PER_TURN, CHAT_HISTORY_LIMIT], sort_keys=True).encode()).hexdigest()[:12]
print(f"Judges ready: conversation (1 call per turn) + grounding (1 call per turn citing documents) · "
      f"{LLM_MODEL} · judge_config_id={JUDGE_CONFIG_ID}")

# COMMAND ----------

# DBTITLE 1,Inputs — assistant turns to score, with thread, next user message, votes
from pyspark.sql import functions as F

src_cols = set(spark.table(SOURCE_TABLE).columns)
TRACE_COL = next((c for c in TRACE_ID_CANDIDATES if c in src_cols), None)
print(f"Trace id column: {TRACE_COL or 'none (MLflow per-trace feedback disabled)'}")
if LOG_FEEDBACK_TO_TRACES and not TRACE_COL:
    LOG_FEEDBACK_TO_TRACES = False

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

# Turns already scored successfully are skipped; turns whose judge failed (turn_verdict NULL) are retried.
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
    df_pairs = df_pairs.withColumn("feedback_vote", F.lit(None).cast("string")) \
                       .withColumn("feedback_comment", F.lit(None).cast("string"))

if SAMPLE_RATE < 1.0:
    df_pairs = df_pairs.filter((F.abs(F.hash(F.col("message_id").cast("string"))) % 1000) < int(SAMPLE_RATE * 1000))

cap = TEST_LIMIT or MAX_TURNS_PER_RUN
df_pairs = df_pairs.orderBy(F.col("created_at").desc()).limit(cap)
MESSAGE_ID_TYPE = df_pairs.schema["message_id"].dataType
CREATED_AT_TYPE = df_pairs.schema["created_at"].dataType

pdf_pairs = df_pairs.toPandas()
print(f"{len(pdf_pairs)} assistant turn(s) to score (cap {cap}).")

# COMMAND ----------

# DBTITLE 1,Turn preparation, deterministic checks, grounding excerpts
import hashlib


def to_messages(prior) -> list:
    """COLLECT_LIST(STRUCT(...)) comes back as dicts or Rows depending on the Spark Connect path."""
    out = []
    for m in prior if prior is not None else []:
        if hasattr(m, "asDict"):
            m = m.asDict()
        out.append({"role": m["role"], "content": m["content"] or ""})
    return out


def trim_history(messages: list, limit: int = CHAT_HISTORY_LIMIT) -> list:
    """Mirrors server/routers/chat.py::_trim_history: keep the last `limit` messages, then drop
    leading turns until the window opens on a user message — the context the agent actually saw."""
    if limit <= 0 or len(messages) <= limit:
        return messages
    trimmed = messages[-limit:]
    while len(trimmed) > 1 and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return trimmed


def last_user_question(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def format_thread(messages: list, per_message_chars: int = 1200, recent_chars: int = 4000) -> str:
    """Older messages are truncated; the two most recent ones are kept almost whole because follow-up
    requests ("remove document X", "same for Y") can only be judged against them."""
    n = len(messages)
    return "\n".join(f"{m['role'].upper()}: {m['content'][:recent_chars if i >= n - 2 else per_message_chars]}"
                     for i, m in enumerate(messages))


_REFUSAL_PHRASES = ["i don't know", "i cannot help", "i'm not sure", "no information available",
                    "je ne sais pas", "je n'ai pas d'information", "aucune information disponible",
                    "impossible de répondre", "je ne peux pas répondre"]
_ALTERNATIVE_HINTS = ["however", "instead", "try", "suggest", "contact",
                      "cependant", "essayez", "contactez", "suggère", "je vous invite"]


def response_length_check(answer: str, answer_type) -> tuple:
    """At least 20 words, except for refusals and clarification requests, which are legitimately short."""
    n = len(str(answer).split())
    ok = n >= 20 or answer_type in ("not_found", "out_of_scope_refusal", "clarification_request")
    return ok, f"{n} words. {'OK' if ok else 'Too short.'}"


def no_empty_refusal(answer: str) -> tuple:
    """A refusal must come with guidance (an alternative, a contact, a suggestion)."""
    text = str(answer).lower()
    refusal = any(p in text for p in _REFUSAL_PHRASES)
    alternative = any(w_ in text for w_ in _ALTERNATIVE_HINTS)
    ok = not refusal or alternative
    return ok, "Refusal + alternative" if refusal and alternative else ("OK" if ok else "Empty refusal.")


from contextlib import nullcontext

from mlflow.entities import AssessmentSource, AssessmentSourceType, Document


def span(name: str, span_type: str):
    """MLflow span when scoring traces are enabled, no-op otherwise."""
    return mlflow.start_span(name=name, span_type=span_type) if LOG_SCORING_TRACES else nullcontext()


def traced_judge(step: str, prompt: str, props: dict, max_tokens: int):
    """Judge call recorded as an LLM step of the turn's trace (prompt, parsed verdict, tokens)."""
    with span(step, "LLM") as sp:
        parsed, usage, err = call_judge(prompt, step, props, max_tokens)
        if sp is not None:
            sp.set_inputs({"model": LLM_MODEL, "prompt": prompt})
            sp.set_outputs(parsed if parsed is not None else {"error": err})
            sp.set_attributes({"input_tokens": usage["in"], "output_tokens": usage["out"], "http_attempts": usage["calls"]})
    return parsed, usage, err


def fetch_excerpts(question: str, answer: str, refs: list) -> tuple:
    """Chunks of the cited documents most related to the question AND the answer's content.
    Returns (excerpts_text, refs_found)."""
    real = resolve_refs(refs)
    if not (VS_OK and real):
        return "", []
    query = f"{question}\n{answer[:800]}"
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX, columns=VS_COLUMNS, query_text=query[:2000], query_type="HYBRID",
        num_results=EXCERPTS_PER_TURN, filters_json=json.dumps({"REF": real}))
    cols = [c.name for c in res.manifest.columns]
    rows = [dict(zip(cols, r)) for r in ((res.result.data_array if res.result else None) or [])]
    blocks, found, docs = [], [], []
    for i, r in enumerate(rows):
        found.append(r.get("REF"))
        head = str(r.get("semantic_headers") or "")[:150]
        text = str(r.get("chunk_text") or "")[:EXCERPT_MAX_CHARS]
        blocks.append(f"[REF: {r.get('REF')} | {head}]\n{text}")
        docs.append(Document(id=f"{r.get('REF')}#{i}", page_content=text, metadata={"doc_uri": r.get("REF"), "section": head}))
    return "\n---\n".join(blocks), sorted(set(filter(None, found))), docs


def grounding_score(claims) -> tuple:
    """(score | None, n_supported, n_partial, n_not_supported, n_not_verifiable). Score over verifiable claims only."""
    counts = {"supported": 0, "partially_supported": 0, "not_supported": 0, "not_verifiable": 0}
    for c in claims or []:
        v = (c or {}).get("verdict")
        if v in counts:
            counts[v] += 1
    checkable = counts["supported"] + counts["partially_supported"] + counts["not_supported"]
    score = (counts["supported"] + 0.5 * counts["partially_supported"]) / checkable if checkable else None
    return score, counts["supported"], counts["partially_supported"], counts["not_supported"], counts["not_verifiable"]


def in_calibration_sample(message_id) -> bool:
    """Stable pseudo-random sample (same turns selected at every run)."""
    h = int(hashlib.md5(str(message_id).encode()).hexdigest()[:8], 16)
    return (h % 10000) < HUMAN_REVIEW_SAMPLE_RATE * 10000

# COMMAND ----------

# DBTITLE 1,Turn verdict — actionable rules on top of the judges
def turn_verdict(r: dict) -> tuple:
    """Returns (verdict, failure_reasons). bad = the user was badly served; acceptable = minor issue."""
    bad, warn = [], []
    at, scope = r.get("answer_type"), r.get("in_scope")

    if r.get("safety__value") is False:
        bad.append("unsafe")
    if at == "error_or_empty":
        bad.append("empty_or_error")
    if r.get("relevance__value") is False:
        bad.append("off_topic")
    if at == "out_of_scope_refusal" and scope == "yes":
        bad.append("wrongful_refusal")
    if r.get("missed_answer"):
        bad.append("missed_answer_in_sources")

    gs, checkable = r.get("groundedness_score"), r.get("claims_checkable") or 0
    n_contradicted = r.get("claims_not_supported") or 0
    if gs is not None:
        if n_contradicted >= 2 or (gs < 0.5 and checkable >= 2):
            bad.append("unsupported_claims")
        elif n_contradicted == 1 or gs < GROUNDED_THRESHOLD:
            warn.append("partially_supported_claims")
    elif r.get("grounding_source") == "plausibility" and r.get("plausibly_grounded") is False:
        bad.append("looks_fabricated")

    level = r.get("completeness_level")
    if at in ("answered", "partial_answer") and level == "none":
        bad.append("does_not_answer")
    elif level == "partial" or at == "partial_answer":
        warn.append("incomplete")

    if r.get("unverified_refs"):
        warn.append("unverified_reference")
    if r.get("language_match__value") is False:
        warn.append("language_mismatch")
    if at in ("answered", "partial_answer") and scope == "yes" and not r.get("citation_count"):
        warn.append("no_citation")
    if r.get("next_turn_signal") == "correction_or_complaint":
        warn.append("user_complaint")
    elif r.get("next_turn_signal") == "rephrase_same_question":
        warn.append("user_rephrased")

    verdict = "bad" if bad else ("acceptable" if warn else "good")
    return verdict, bad + warn

# COMMAND ----------

# DBTITLE 1,Score one turn (quality call + grounding call), deterministic metrics, verdict
def score_turn(row) -> dict:
    """Scores one turn. With LOG_SCORING_TRACES, the whole evaluation is one MLflow trace (inputs = conversation,
    outputs = assistant answer, steps = judges and cited excerpts) and every verdict is attached to it."""
    if not LOG_SCORING_TRACES:
        return _score_turn(row, None)
    with mlflow.start_span(name="turn_evaluation", span_type="CHAIN") as root:
        rec = _score_turn(row, root)
    rec["scoring_trace_id"] = root.trace_id
    try:
        log_verdicts(root.trace_id, rec)
    except Exception as e:
        rec["judge_errors"].append(f"verdict logging: {str(e)[:200]}")
    return rec


def _score_turn(row, root) -> dict:
    thread = trim_history(to_messages(row["prior_messages"]))
    question = last_user_question(thread)
    answer = str(row["answer"] or "")
    src = source_refs(row["sources_json"])
    cited = sorted(code_like(answer))
    next_msg = row.get("next_user_message")
    next_msg = str(next_msg)[:800] if next_msg is not None and str(next_msg) != "nan" else "(none)"

    rec = {
        "message_id": row["message_id"], "created_at": row["created_at"], "session_id": row["session_id"],
        "division": row["division"], "endpoint_name": row["endpoint_name"], "trace_id": row.get("trace_id"),
        "user_question": question, "thread_turn_count": len(thread), "answer": answer[:2000],
        "next_user_message": None if next_msg == "(none)" else next_msg,
        "citation_count": len(src), "source_refs": src, "cited_refs": cited,
        "approximate_refs": [], "unindexed_refs": [], "unverified_refs": [],
        "feedback_vote": row.get("feedback_vote"), "feedback_comment": row.get("feedback_comment"),
        "judge_model": LLM_MODEL, "judge_config_id": JUDGE_CONFIG_ID, "judge_errors": [], "scoring_trace_id": None,
    }
    tin = tout = calls = 0
    if root is not None:
        root.set_inputs({"conversation": thread, "sources_listed_by_assistant": src,
                         "next_user_message": rec["next_user_message"]})
        root.set_outputs({"answer": answer})
        mlflow.update_current_trace(
            tags={"message_id": str(row["message_id"]), "endpoint": str(row["endpoint_name"]),
                  "division": str(row["division"]), "ka_trace_id": str(row.get("trace_id") or ""),
                  "judge_config_id": JUDGE_CONFIG_ID},
            metadata={"mlflow.trace.session": str(row["session_id"])})

    # 1) Conversation-level quality (one structured call)
    q_prompt = QUALITY_PROMPT.format(thread=format_thread(thread), answer=answer[:6000],
                                     sources=", ".join(src) or "(none)", next_message=next_msg)
    q, u, err = traced_judge("conversation_judge", q_prompt, QUALITY_PROPS, MAX_TOKENS_QUALITY)
    tin, tout, calls = tin + u["in"], tout + u["out"], calls + u["calls"]
    q = q or {}
    if err:
        rec["judge_errors"].append(f"quality: {err}")
    rec.update({
        "question_intent": q.get("question_intent"), "question_topic": q.get("question_topic"),
        "question_language": q.get("question_language"), "in_scope": q.get("in_scope"),
        "is_follow_up": q.get("is_follow_up"), "answer_type": q.get("answer_type"),
        "relevance__value": q.get("relevance_pass"), "relevance__rationale": q.get("relevance_rationale"),
        "completeness_level": q.get("completeness_level"),
        "completeness__value": (q.get("completeness_level") in ("full", "not_applicable")) if q else None,
        "completeness__rationale": q.get("completeness_rationale"),
        "language_match__value": q.get("language_match_pass"), "language_match__rationale": q.get("language_match_rationale"),
        "safety__value": q.get("safety_pass"), "safety__rationale": q.get("safety_rationale"),
        "plausibly_grounded": q.get("plausibly_grounded_pass"),
        "next_turn_signal": q.get("next_turn_signal"), "next_turn_rationale": q.get("next_turn_rationale"),
    })

    # 2) Grounding against the cited documents (when there is something to verify)
    rec.update({"grounding_source": "none", "groundedness_score": None, "claims_supported": None,
                "claims_partial": None, "claims_not_supported": None, "claims_not_verifiable": None,
                "claims_checkable": None, "unsupported_claims": [], "context_sufficient": None,
                "missed_answer": None, "missed_answer_detail": None, "grounding_rationale": None,
                "excerpt_refs": []})
    refs_to_check = list(dict.fromkeys(src + cited))
    excerpts = ""
    if q and rec["answer_type"] in ("answered", "partial_answer", "not_found") and refs_to_check and VS_OK:
        with span("cited_document_excerpts", "RETRIEVER") as sp:
            try:
                excerpts, found, docs = fetch_excerpts(question, answer, refs_to_check)
            except Exception as e:
                excerpts, found, docs = "", [], []
                rec["judge_errors"].append(f"vector_search: {str(e)[:200]}")
            if sp is not None:
                sp.set_inputs({"query": question, "cited_documents": refs_to_check})
                sp.set_outputs(docs)
        if excerpts:
            g_prompt = GROUNDING_PROMPT.format(question=question, answer=answer[:6000], excerpts=excerpts)
            g, u, err = traced_judge("grounding_judge", g_prompt, GROUNDING_PROPS, MAX_TOKENS_GROUNDING)
            tin, tout, calls = tin + u["in"], tout + u["out"], calls + u["calls"]
            if err:
                rec["judge_errors"].append(f"grounding: {err}")
            if g:
                score, sup, part, nsup, nver = grounding_score(g.get("claims"))
                rec.update({
                    "grounding_source": "cited_documents", "groundedness_score": score,
                    "claims_supported": sup, "claims_partial": part, "claims_not_supported": nsup,
                    "claims_not_verifiable": nver, "claims_checkable": sup + part + nsup,
                    "unsupported_claims": [str(c.get("claim"))[:300] for c in g.get("claims") or []
                                           if c.get("verdict") == "not_supported"],
                    "context_sufficient": g.get("context_sufficient"),
                    "missed_answer": bool(g.get("missed_information")),
                    "missed_answer_detail": (g.get("missed_information_detail") or None),
                    "grounding_rationale": g.get("rationale"), "excerpt_refs": found,
                })

    rec["approximate_refs"], rec["unindexed_refs"], rec["unverified_refs"] = classify_refs(cited, excerpts)

    # groundedness__value: verified score when claims could be checked, otherwise the plausibility check
    if rec["groundedness_score"] is not None:
        rec["groundedness__value"] = rec["groundedness_score"] >= GROUNDED_THRESHOLD
        rec["groundedness__rationale"] = rec["grounding_rationale"]
    else:
        if q and rec["grounding_source"] == "none":
            rec["grounding_source"] = "plausibility"
        rec["groundedness__value"] = rec["plausibly_grounded"]
        rec["groundedness__rationale"] = q.get("plausibly_grounded_rationale")

    # 3) Deterministic checks (free)
    rec["response_length_check__value"], rec["response_length_check__rationale"] = response_length_check(answer, rec["answer_type"])
    rec["no_empty_refusal__value"], rec["no_empty_refusal__rationale"] = no_empty_refusal(answer)

    # 4) Verdict, review queue, cost
    if q:
        rec["turn_verdict"], rec["failure_reasons"] = turn_verdict(rec)
    else:
        rec["turn_verdict"], rec["failure_reasons"] = None, ["judge_failed"]
    disagreement = ((rec["turn_verdict"] == "good" and rec["feedback_vote"] == "down")
                    or (rec["turn_verdict"] == "bad" and rec["feedback_vote"] == "up"))
    rec["needs_human_review"] = bool(rec["turn_verdict"]) and bool(disagreement or in_calibration_sample(rec["message_id"]))
    rec["review_reason"] = ("judge_vs_user_disagreement" if disagreement
                            else "calibration_sample" if rec["needs_human_review"] else None)
    rec["golden_candidate"] = rec["turn_verdict"] == "bad" or rec["feedback_vote"] == "down"
    rec["total_input_tokens"], rec["total_output_tokens"], rec["n_llm_calls"] = tin, tout, calls
    rec["cost_usd"] = round(cost_usd(tin, tout), 6)
    if root is not None:
        mlflow.update_current_trace(tags={"turn_verdict": str(rec["turn_verdict"])})
    return rec


JUDGE_SOURCE = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=LLM_MODEL)
USER_SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="chat_user")


def log_verdicts(trace_id: str, rec: dict):
    """Attaches every judgment of the turn to its scoring trace (visible in the trace's Assessments panel)."""
    refs_note = "; ".join(x for x in [
        f"resolved: {rec['approximate_refs']}" if rec["approximate_refs"] else "",
        f"outside the corpus: {rec['unindexed_refs']}" if rec["unindexed_refs"] else "",
        f"unverified: {rec['unverified_refs']}" if rec["unverified_refs"] else ""] if x)
    items = [
        ("turn_verdict", rec["turn_verdict"], ", ".join(rec["failure_reasons"]) or "no issue"),
        ("answer_type", rec["answer_type"], None),
        ("question_intent", rec["question_intent"], rec["question_topic"]),
        ("relevance", rec["relevance__value"], rec["relevance__rationale"]),
        ("completeness", rec["completeness_level"], rec["completeness__rationale"]),
        ("groundedness", rec["groundedness_score"], rec["grounding_rationale"]),
        ("unsupported_claims", len(rec["unsupported_claims"]) if rec["grounding_source"] == "cited_documents" else None,
         " | ".join(rec["unsupported_claims"]) or None),
        ("missed_answer", rec["missed_answer"], rec["missed_answer_detail"]),
        ("reference_integrity", not rec["unverified_refs"], refs_note or "all cited codes exist"),
        ("language_match", rec["language_match__value"], rec["language_match__rationale"]),
        ("safety", rec["safety__value"], rec["safety__rationale"]),
        ("next_turn_signal", rec["next_turn_signal"], rec["next_turn_rationale"]),
    ]
    for name, value, why in items:
        if value is not None:
            mlflow.log_feedback(trace_id=trace_id, name=name, value=value, rationale=why or None, source=JUDGE_SOURCE)
    if rec["feedback_vote"] in ("up", "down"):
        mlflow.log_feedback(trace_id=trace_id, name="user_vote", value=rec["feedback_vote"],
                            rationale=rec["feedback_comment"] or None, source=USER_SOURCE)

# COMMAND ----------

# DBTITLE 1,Output schema
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, LongType, StringType,
                               StructField, StructType)

# Counts are BIGINT; the schema is explicit so that all-NULL columns never break type inference
S, B, I, D, A = StringType(), BooleanType(), LongType(), DoubleType(), ArrayType(StringType())
OUT_SCHEMA = StructType([
    StructField("message_id", MESSAGE_ID_TYPE), StructField("created_at", CREATED_AT_TYPE),
    StructField("session_id", S), StructField("division", S), StructField("endpoint_name", S), StructField("trace_id", S),
    StructField("user_question", S), StructField("thread_turn_count", I), StructField("answer", S),
    StructField("next_user_message", S), StructField("citation_count", I),
    StructField("source_refs", A), StructField("cited_refs", A), StructField("approximate_refs", A), StructField("unindexed_refs", A), StructField("unverified_refs", A), StructField("excerpt_refs", A),
    StructField("feedback_vote", S), StructField("feedback_comment", S),
    # classification
    StructField("question_intent", S), StructField("question_topic", S), StructField("question_language", S),
    StructField("in_scope", S), StructField("is_follow_up", B), StructField("answer_type", S),
    # boolean quality dimensions used by the dashboard
    StructField("relevance__value", B), StructField("relevance__rationale", S),
    StructField("groundedness__value", B), StructField("groundedness__rationale", S),
    StructField("safety__value", B), StructField("safety__rationale", S),
    StructField("language_match__value", B), StructField("language_match__rationale", S),
    StructField("completeness__value", B), StructField("completeness__rationale", S),
    StructField("response_length_check__value", B), StructField("response_length_check__rationale", S),
    StructField("no_empty_refusal__value", B), StructField("no_empty_refusal__rationale", S),
    # detailed judgments
    StructField("completeness_level", S), StructField("plausibly_grounded", B),
    StructField("next_turn_signal", S), StructField("next_turn_rationale", S),
    StructField("grounding_source", S), StructField("groundedness_score", D),
    StructField("claims_supported", I), StructField("claims_partial", I), StructField("claims_not_supported", I),
    StructField("claims_not_verifiable", I), StructField("claims_checkable", I), StructField("unsupported_claims", A),
    StructField("context_sufficient", B), StructField("missed_answer", B), StructField("missed_answer_detail", S),
    StructField("grounding_rationale", S),
    StructField("turn_verdict", S), StructField("failure_reasons", A),
    StructField("needs_human_review", B), StructField("review_reason", S), StructField("golden_candidate", B),
    StructField("judge_model", S), StructField("judge_config_id", S), StructField("judge_errors", A),
    StructField("scoring_trace_id", S),
    StructField("total_input_tokens", I), StructField("total_output_tokens", I), StructField("n_llm_calls", I),
    StructField("cost_usd", D), StructField("scored_at", S),
])
OUT_COLS = [f.name for f in OUT_SCHEMA.fields]


def _clean(v, field):
    """Python value → value accepted by the Spark field type (NaN/None handling, numpy scalars, casts)."""
    if hasattr(v, "item") and not isinstance(v, (str, bytes, list, dict)) and getattr(v, "ndim", 0) == 0:
        v = v.item()                          # numpy.int64 / numpy.bool_ → Python scalars
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()                        # numpy arrays → lists
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


def merge_batch(records: list):
    rows = [tuple(_clean(r.get(f.name), f) for f in OUT_SCHEMA.fields) for r in records]
    sdf = spark.createDataFrame(rows, OUT_SCHEMA)
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

# COMMAND ----------

# DBTITLE 1,Run — parallel scoring by batch, MERGE after each batch (crash-safe)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

RUN_TS = datetime.now(timezone.utc).isoformat(timespec="seconds")
t_start = time.time()
all_records = []

if pdf_pairs.empty:
    print("Nothing new to score.")
elif DRY_RUN:
    # Estimate only: prompt sizes are real, outputs and grounding share are assumptions
    est_in = 0
    for _, row in pdf_pairs.iterrows():
        thread = trim_history(to_messages(row["prior_messages"]))
        est_in += len(QUALITY_PROMPT) + len(format_thread(thread)) + min(len(str(row["answer"])), 6000)
    est_in_tok = est_in / 3.8
    n_ground = int(len(pdf_pairs) * 0.7)
    est_in_tok += n_ground * (len(GROUNDING_PROMPT) + EXCERPTS_PER_TURN * 1200 + 3000) / 3.8
    est_out_tok = len(pdf_pairs) * 1200 + n_ground * 1500
    print(f"DRY RUN (nothing is scored or written) · {len(pdf_pairs)} turns · ~{len(pdf_pairs) + n_ground} calls · "
          f"~{est_in_tok / 1e6:.2f}M in / ~{est_out_tok / 1e6:.2f}M out tokens · ≈ ${cost_usd(est_in_tok, est_out_tok):.2f}")
else:
    rows = [r for _, r in pdf_pairs.iterrows()]
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        records = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(score_turn, r): r["message_id"] for r in batch}
            for fut in as_completed(futures):
                try:
                    records.append(fut.result())
                except Exception as e:   # never lose the batch because of one turn
                    print(f"⚠️ turn {futures[fut]} failed: {str(e)[:200]}")
        for r in records:
            r["scored_at"] = RUN_TS
        if records:
            merge_batch(records)
            all_records.extend(records)
        spent = sum(r["cost_usd"] for r in all_records)
        print(f"… {min(start + BATCH_SIZE, len(rows))}/{len(rows)} turns · ${spent:.3f} · {time.time() - t_start:.0f}s")

df_final = pd.DataFrame(all_records)
if not df_final.empty:
    print("\nVerdicts:", df_final["turn_verdict"].value_counts(dropna=False).to_dict())
    print("Answer types:", df_final["answer_type"].value_counts(dropna=False).to_dict())
    print("Grounding source:", df_final["grounding_source"].value_counts(dropna=False).to_dict())
    print(f"Tokens: {df_final['total_input_tokens'].sum():,} in / {df_final['total_output_tokens'].sum():,} out · "
          f"${df_final['cost_usd'].sum():.3f} · {len(df_final[df_final['judge_errors'].map(len) > 0])} turn(s) with judge errors")

# COMMAND ----------

# DBTITLE 1,Run ledger — exact tokens and cost, quality summary, judge vs user agreement
def rate(series, value=True):
    s = series.dropna()
    return round(float((s == value).mean()), 4) if len(s) else None


run_row = None
if not df_final.empty:
    voted = df_final[df_final["feedback_vote"].isin(["up", "down"]) & df_final["turn_verdict"].notna()]
    agree = None
    if len(voted):
        agree = round(float(((voted["turn_verdict"] != "bad") == (voted["feedback_vote"] == "up")).mean()), 4)
    grounded = df_final[df_final["grounding_source"] == "cited_documents"]["groundedness_score"].dropna()
    run_row = {
        "run_ts": RUN_TS, "judge_model": LLM_MODEL, "judge_config_id": JUDGE_CONFIG_ID,
        "n_messages": int(len(df_final)), "n_llm_calls": int(df_final["n_llm_calls"].sum()),
        "total_input_tokens": int(df_final["total_input_tokens"].sum()),
        "total_output_tokens": int(df_final["total_output_tokens"].sum()),
        "estimated_cost_usd": round(float(df_final["cost_usd"].sum()), 6),
        "duration_s": round(time.time() - t_start, 1),
        "n_judge_errors": int((df_final["judge_errors"].map(len) > 0).sum()),
        "good_rate": rate(df_final["turn_verdict"], "good"),
        "bad_rate": rate(df_final["turn_verdict"], "bad"),
        "refusal_rate": rate(df_final["answer_type"].isin(["not_found", "out_of_scope_refusal"]).where(df_final["answer_type"].notna())),
        "groundedness_mean": round(float(grounded.mean()), 4) if len(grounded) else None,
        "grounding_coverage": rate(df_final["grounding_source"] == "cited_documents"),
        "judge_user_agreement": agree, "n_voted": int(len(voted)),
        "n_needs_human_review": int(df_final["needs_human_review"].sum()),
    }
    RUNS_SCHEMA = StructType([StructField(k, t) for k, t in [
        ("run_ts", S), ("judge_model", S), ("judge_config_id", S), ("n_messages", I), ("n_llm_calls", I),
        ("total_input_tokens", I), ("total_output_tokens", I), ("estimated_cost_usd", D), ("duration_s", D),
        ("n_judge_errors", I), ("good_rate", D), ("bad_rate", D), ("refusal_rate", D), ("groundedness_mean", D),
        ("grounding_coverage", D), ("judge_user_agreement", D), ("n_voted", I), ("n_needs_human_review", I)]])
    # explicit schema: an all-NULL metric (e.g. no vote yet) must not break type inference
    spark.createDataFrame([tuple(_clean(run_row[f.name], f) for f in RUNS_SCHEMA.fields)], RUNS_SCHEMA) \
         .write.mode("append").option("mergeSchema", "true").saveAsTable(SCORING_RUNS_TABLE)
    print(json.dumps(run_row, indent=1, default=str))

# COMMAND ----------

# DBTITLE 1,Alerts — today's bad rate vs 7-day baseline, per agent
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

# DBTITLE 1,MLflow — one run per scoring run (trend charts + worst turns), optional feedback on KA traces
MLFLOW_RUN_ID = None
if LOG_MLFLOW_RUN and run_row:
    try:
        with mlflow.start_run(run_name=f"quality-scoring {RUN_TS}") as _run:
            MLFLOW_RUN_ID = _run.info.run_id
            mlflow.set_tags({"judge_config_id": JUDGE_CONFIG_ID, "judge_model": LLM_MODEL, "alerts": " | ".join(alerts)[:4000]})
            mlflow.log_metrics({k: float(v) for k, v in run_row.items()
                                if isinstance(v, (int, float)) and v is not None and k != "run_ts"})
            reasons = df_final.explode("failure_reasons")["failure_reasons"].value_counts()
            mlflow.log_metrics({f"reason/{k}": int(v) for k, v in reasons.items() if isinstance(k, str)})
            by_agent = df_final.groupby("endpoint_name")["turn_verdict"].apply(lambda s: (s == "bad").mean())
            mlflow.log_metrics({f"bad_rate/{k}": float(v) for k, v in by_agent.items()})
            worst = df_final[df_final["turn_verdict"] == "bad"].head(50)
            if len(worst):
                mlflow.log_table(worst[["message_id", "endpoint_name", "question_intent", "user_question", "answer",
                                        "failure_reasons", "unsupported_claims", "missed_answer_detail",
                                        "feedback_vote"]].astype(str), "worst_turns.json")
        print("✓ MLflow run logged")
    except Exception as e:
        print(f"⚠️ MLflow run not logged: {str(e)[:200]}")

if LOG_FEEDBACK_TO_TRACES and not df_final.empty:
    from mlflow.entities import AssessmentSource, AssessmentSourceType
    src = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=f"{LLM_MODEL}/{JUDGE_CONFIG_ID}")
    ok = ko = 0
    for r in df_final[df_final["trace_id"].notna() & df_final["turn_verdict"].notna()].to_dict("records"):
        try:
            mlflow.log_feedback(trace_id=r["trace_id"], name="qualibot_turn_verdict", value=r["turn_verdict"],
                                rationale=", ".join(r["failure_reasons"]) or "no issue", source=src)
            ok += 1
        except Exception:
            ko += 1   # typically: trace in another workspace/experiment without CAN_EDIT
    print(f"Feedback on KA traces: {ok} ok · {ko} failed")

if FAIL_JOB_ON_ALERT and alerts:
    raise RuntimeError("Quality alert: " + " | ".join(alerts))

# COMMAND ----------

# DBTITLE 1,Run summary — what was written, and where to look
if DRY_RUN:
    print("Dry run: no judge call, no table write, no MLflow run. Set dry_run=false to score.")
elif df_final.empty:
    print("No new turn to score: tables and MLflow are unchanged.")
else:
    n_total = spark.table(SCORES_TABLE).count()
    print(f"Scored turns written : {len(df_final)} → {SCORES_TABLE} ({n_total} rows in total)")
    print(f"Run ledger           : 1 row → {SCORING_RUNS_TABLE}")
    print(f"MLflow               : {EVAL_EXPERIMENT_PATH} → Traces tab (one trace per turn), run {MLFLOW_RUN_ID or 'not logged'}")
    print(f"Judge configuration  : {JUDGE_CONFIG_ID}")
    display(spark.table(SCORES_TABLE).filter(F.col("scored_at") == RUN_TS)
            .select("created_at", "endpoint_name", "question_intent", "answer_type", "turn_verdict",
                    "failure_reasons", "groundedness_score", "user_question")
            .orderBy(F.col("created_at").desc()))

# COMMAND ----------

# DBTITLE 1,Dashboard queries (copy into the Lakeview "ChatBot - Quality" page)
# MAGIC %md
# MAGIC ```sql
# MAGIC -- 1. Daily verdicts per agent
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
# MAGIC        AVG(groundedness_score) AS groundedness,
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
# MAGIC -- 7. Cost per day
# MAGIC SELECT DATE(run_ts) AS day, SUM(estimated_cost_usd) AS usd, SUM(n_messages) AS turns
# MAGIC FROM uat_proj.qualibot.chat_quality_scoring_runs GROUP BY ALL ORDER BY day;
# MAGIC ```
