# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Qualibot — Golden Evaluation Dataset Builder
# MAGIC
# MAGIC Builds the golden evaluation dataset used to evaluate the Qualibot Knowledge Assistants
# MAGIC (`utils/evaluation/Evaluate_Knowledge_Assistant.py`). Every reference answer is built from evidence retrieved in
# MAGIC the document index and checked by several LLM judges, so that only a light human review is needed.
# MAGIC
# MAGIC | # | Step | LLM calls |
# MAGIC |---|---|---|
# MAGIC | 1 | Extract question/answer turns from the chat logs, with conversation history and user votes | – |
# MAGIC | 2 | Annotate every question: intent, quality, test value, self-contained rewrite, feedback triage | 1 per question |
# MAGIC | 3 | Build a diversified shortlist: embeddings + clustering + quotas, including production failures | embeddings |
# MAGIC | 4 | Query the current assistant (answer and the passages it retrieved, from its trace), and generate search queries (keywords, translation, title, hypothetical answer) | 1 per case |
# MAGIC | 5 | Pool evidence from many retrieval routes | – |
# MAGIC | 6 | Grade every candidate chunk, expand around the most relevant ones (similar chunks of the same document, and the chunks just before and after in document order), grade the additions | 1 per chunk |
# MAGIC | 7 | Generate two independent reference answers (A/B) from the relevant chunks | 2 per case |
# MAGIC | 8 | Arbitrate: merge A/B, confront the assistant's answers, verdicts, confidence | 1 per case |
# MAGIC | 9 | Verify every fact independently | 1 per case |
# MAGIC | 10 | Consolidate and select a balanced final set of 20 to 30 cases, compliance-matrix questions included | – |
# MAGIC | 11 | Human review (validate / reject / expert) | – |
# MAGIC | 12 | Export to the MLflow evaluation dataset (Unity Catalog), linked to the evaluation experiment, and to the flat table `ka_eval_golden_cases` (composition and review status, for the dashboard) | – |
# MAGIC | 13 | Cost: estimate vs actual | – |
# MAGIC
# MAGIC **Persistence and restart.** Every step is cached in a single Delta table (`CACHE_TABLE`, one row per
# MAGIC step × question [× chunk], JSON payload). Each section reloads its inputs from the cache: run section 0, then any
# MAGIC section directly. LLM steps only process rows that are missing or failed, so re-running never pays twice.
# MAGIC To recompute a step: `FORCE = {"step_name"}`, run it, then reset `FORCE = set()`.
# MAGIC
# MAGIC **Document references.** Codes are compared with a key insensitive to language suffix (`PRLAT538_FR` = `PRLAT538_GB`),
# MAGIC separators, case and zero padding (`IN_APO_006` = `IN_APO_0006`, a typo that exists inside some documents).

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

NEEDED = {"mlflow": (3, 11), "scikit-learn": (1, 0)}   # same MLflow as the evaluation notebook


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
            if "==" in l and not l.lower().startswith(("mlflow", "scikit-learn"))]
    with open("/tmp/pinned_packages.txt", "w") as f:
        f.write("\n".join(pins))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/pinned_packages.txt", *missing])
    dbutils.library.restartPython()
print({p: _installed(p) for p in NEEDED})

# COMMAND ----------

# DBTITLE 1,0.1 Configuration
import hashlib
import html as html_mod
import json
import math
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import types as T

# ── Sources and storage ──
SRC = "uat_landingzone.qualibot"                       # chat_messages, chat_feedbacks
CACHE_TABLE = f"{SRC}.qualibot_eval_cache"             # cache of every step
EVAL_DATASET_UC = f"{SRC}.qualibot_eval_golden"        # MLflow evaluation dataset
PROD_SCORES_TABLE = "uat_proj.qualibot.chat_quality_scores"   # production scoring (source of failure cases)
GOLDEN_CASES_TABLE = "uat_proj.qualibot.ka_eval_golden_cases"  # flat view of the reviewed cases, for the dashboard
VS_ENDPOINT = "qualibot"
VS_INDEX = f"{SRC}.chunks_index_v1"
VS_COLUMNS = ["REF", "chunk_text", "semantic_headers"]
REF_SOURCE_TABLE = None                                # source table of the index; None = read from the index

# ── Models and endpoints ──
JUDGE_MODEL = "databricks-gpt-6-luna"                 # called directly: the endpoint does not support batch inference
EMBED_MODEL = "databricks-gte-large-en"                # only used for diversity and de-duplication (ai_query)
LLM_WORKERS = 8                                        # structured calls in flight
LLM_RATE_SHARE = 0.7                                   # share of the judge model's token limits used by this notebook
LLM_INPUT_TOKENS_PER_MINUTE = 200_000
LLM_OUTPUT_TOKENS_PER_MINUTE = 20_000
LLM_MAX_ATTEMPTS = 6                                   # retries of a rate-limited or unavailable call (2 s … 60 s)
LLM_KEYS_PER_WRITE = 300                               # rows of a step written to the cache at a time
KA_ENDPOINT = "ka-7679a56e-endpoint"                   # qualibot_ALL_v2: the assistant evaluated
KA_MAX_CONCURRENT = 3                                  # capacity limit of the assistant endpoint
QUERY_KA_FRESH = True                                  # query the current assistant for every shortlisted case

# ── Users excluded from the logs ──
EXCLUDED_USER_IDS = {"8216413032099407", "71172701575508", "73401798024381"}
INCLUDE_EMAIL = "jules.gourio.external@latecoere.aero"
EXCLUDED_GROUPS = {"Role-Project-LEAP-CoreDev", "Role-Project-LEAP-CoreAdmin"}

# ── Sizes and thresholds ──
TARGET_N = 30                    # final dataset size (20 to 30 cases after the human review)
MIN_EXPORTED = 20                # the export warns below this size
HISTORY_TURNS = 4                # history messages kept for multi-turn questions
N_CLUSTERS = 60
DEDUP_SIM = 0.90                 # cosine above which two questions are duplicates
CLUSTER_PENALTY = 1.5            # penalty when a topic cluster is already represented
MAX_POOL_CHUNKS = 45             # candidate chunks graded per case (first pass)
EXPANSION_PER_CHUNK = 4          # chunks fetched around each highly relevant chunk (second pass)
EXCLUDED_INTENTS = {"chitchat_or_meta", "link_or_navigation"}
LANG_SUFFIXES = ["FR", "GB", "EN", "UK", "CZ", "ES", "DE", "PT", "IT", "MX", "BG", "RO", "PL", "TN"]
NEIGHBOUR_WINDOW = 1             # chunks taken before and after each directly relevant chunk (document order)

# Shortlist quotas for log questions (predicates in section 3)
SLOT_QUOTAS = {
    "production_retrieval_miss": 3, "production_compliance_claim": 3, "production_failure": 4, "production_pass": 6,
    "negative_feedback": 5, "suspect_answer": 4, "multi_doc_or_hard": 4,
    "requirement_compliance": 4, "document_lookup": 5, "procedure": 4, "rule_requirement": 4,
    "definition_acronym": 3, "multi_turn": 3, "out_of_scope": 2, "other_language": 1,
}
# Final selection constraints
MAX_PER_INTENT = 6
MIN_KA_FAIL = 10                 # cases the assistant fails: they discriminate between versions of the assistant
MIN_KA_OK = 8                    # cases the assistant passes: regression protection
MIN_COMPLIANCE = 3               # customer compliance-matrix questions (about half of the production traffic)
MIN_REFUSAL_CASES = 2            # expected answer = "not in the documentation" / out of scope
MAX_NEEDS_EXPERT = 3             # low-confidence cases kept for expert review

# ── Cost (pay-per-token, DBU per 1M tokens) ──
DBU_PER_M_INPUT = 1.4
DBU_PER_M_OUTPUT = 7.1
USD_PER_DBU = 0.07
CHARS_PER_TOKEN = 3.8            # estimate, recalibrated in section 13
OUTPUT_OVERHEAD = 1.0            # > 1 when the model bills hidden reasoning tokens (calibrated in section 13)
DEFAULT_OUT_CHARS = 1500
ESTIMATE_ONLY = set()            # steps to price WITHOUT calling the LLM, e.g. {"chunk_grades"}
USAGE_SINCE = "2026-09-01"

# ── Evaluation experiment (Unity Catalog trace storage) ──
ME_EMAIL = "jules.gourio.external@latecoere.aero"
TRACES_CATALOG, TRACES_SCHEMA = "uat_proj", "qualibot"
EVAL_EXPERIMENT = f"/Workspace/Users/{ME_EMAIL}/qualibot-traces/trace_eval_all_v2"
EVAL_TRACE_PREFIX = "trace_eval_all_v2"
SQL_WAREHOUSE_ID = "5890912c31867b77"

FORCE = set()                    # steps to recompute, e.g. {"reformulations"}; {"ALL"} for everything

# ── Curated cases: hints for the judges (checked against the excerpts) and forced expectations ──
#   expected_response / expected_sources : hints given to the judges
#   human_facts       : facts exported as essential expected facts, whatever the judges found
#   human_guidelines  : guidelines exported with the case
INTRAQUAL_URL = "https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref={ref}"
MANUAL_OVERRIDES = [
    {"question_id": 1881, "expected_sources": "QP-1457",
     "expected_response": "The supplier selection procedure of the purchasing process is described in QP-1457.",
     "human_guidelines": ["Must cite procedure QP-1457."]},
    {"question_id": 293, "expected_sources": "PRLAT508",
     "expected_response": "Laser marking machines are production means; the risk analysis methodology for production "
                          "means is described in PRLAT 508.",
     "human_facts": ["Laser marking machines are categorised as production means.",
                     "The risk analysis methodology for production means is described in PRLAT508."],
     "human_guidelines": ["Must cite document PRLAT508."]},
    {"question_id": 2495, "expected_sources": "QP-1518, MR-1226_EN",
     "human_guidelines": ["Must answer in Spanish.", "Must rely on QP-1518 and/or MR-1226."]},
]
SYNTHETIC_QUERIES = [
    {"question_id": -1, "question": "que signifie l'acronyme APO ?",
     "expected_response": "APO means Analyste Performance Opérationnelle.", "expected_sources": "IN_APO_0006",
     "human_facts": ["APO means Analyste Performance Opérationnelle."]},
    {"question_id": -2, "question": "Donne moi le lien de l'OPEX Sharepoint",
     "expected_response": "Gives the links contained in document INAQ742.", "expected_sources": "INAQ742",
     "human_guidelines": ["Must not invent any SharePoint link: only links found in the documentation are acceptable."]},
    {"question_id": -3, "question": "quelle est la règle concernant les FAI pour des pièces qui n'ont pas été fabriquées depuis plus de 2 ans?",
     "expected_sources": "INAQ619_FR, PRLAT538_FR"},
    {"question_id": -4, "question": "fais moi ma liste de courses pour ce week end",
     "expected_response": "Out of scope: Qualibot only answers questions about the aerospace quality documentation.",
     "human_guidelines": ["Must politely decline the request.",
                          "Must state which kind of questions the assistant answers (quality documentation)."]},
    {"question_id": -5, "question": "Trouve moi le template du CMP",
     "expected_response": "The Configuration Management Plan (CMP) template is document NF-10065 "
                          "« Template Configuration Management Plan ».",
     "expected_sources": "NF-10065",
     "human_facts": ["The Configuration Management Plan (CMP) template is document NF-10065."]},
]
CURATED = {o["question_id"]: o for o in MANUAL_OVERRIDES + SYNTHETIC_QUERIES}
CURATED_IDS = set(CURATED)

# COMMAND ----------

# DBTITLE 1,0.2 Helpers — step cache, structured LLM calls, cost, document references, assistant calls
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

spark.sql(f"""CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
    stage STRING, question_id BIGINT, chunk_id STRING,
    payload STRING, payload_schema STRING, updated_at TIMESTAMP) USING DELTA
    COMMENT 'Golden dataset builder cache: one row per step x question [x chunk], JSON payload'""")

CHUNK_STAGES = {"evidence_pool", "evidence_expansion", "evidence_neighbours", "chunk_grades"}


# ── Step cache ──
def _forced(stage):
    return "ALL" in FORCE or stage in FORCE


def _cache(stage):
    return spark.table(CACHE_TABLE).filter(F.col("stage") == stage)


def load_stage(stage, chunk_level=None):
    """Typed DataFrame of a cached step, or None if the step never ran."""
    chunk_level = stage in CHUNK_STAGES if chunk_level is None else chunk_level
    t = _cache(stage)
    row = t.orderBy(F.desc("updated_at")).select("payload_schema").limit(1).collect()
    if not row:
        return None
    schema = T.StructType.fromJson(json.loads(row[0][0]))
    keys = ["question_id"] + (["chunk_id"] if chunk_level else [])
    return t.select(*keys, F.from_json("payload", schema).alias("_d")).select(*keys, "_d.*")


def stage_payloads(stage) -> list:
    """Raw JSON payloads of a small step (robust to NULL-only fields)."""
    rows = _cache(stage).select("question_id", "payload").collect()
    return [{**json.loads(r.payload), "question_id": int(r.question_id)} for r in rows]


def _append(stage, df):
    data = [c for c in df.columns if c not in ("question_id", "chunk_id")]
    st = df.select(F.struct(*data).alias("s")).schema["s"].dataType
    (df.select(F.lit(stage).alias("stage"),
               F.col("question_id").cast("long").alias("question_id"),
               (F.col("chunk_id").cast("string") if "chunk_id" in df.columns else F.lit("")).alias("chunk_id"),
               F.to_json(F.struct(*data)).alias("payload"),
               F.lit(st.json()).alias("payload_schema"),
               F.current_timestamp().alias("updated_at"))
       .write.mode("append").saveAsTable(CACHE_TABLE))


def _delete_stage(stage):
    spark.sql(f"DELETE FROM {CACHE_TABLE} WHERE stage = '{stage}'")


def _keys_df(rows, chunk_level):
    if chunk_level:
        return spark.createDataFrame([(int(r[0]), str(r[1])) for r in rows], "question_id long, chunk_id string")
    return spark.createDataFrame([(int(r[0]),) for r in rows], "question_id long")


def _delete_keys(stage, kdf, chunk_level=False):
    cond = "t.question_id = s.question_id" + (" AND t.chunk_id = s.chunk_id" if chunk_level else "")
    kdf.createOrReplaceTempView("_del_keys")
    spark.sql(f"MERGE INTO {CACHE_TABLE} t USING _del_keys s ON t.stage = '{stage}' AND {cond} WHEN MATCHED THEN DELETE")


def replace_stage(stage, df):
    """Replaces a whole step. df may read the step itself: written under a temporary name, then switched."""
    tmp = f"{stage}__tmp"
    _delete_stage(tmp)
    _append(tmp, df)
    _delete_stage(stage)
    spark.sql(f"UPDATE {CACHE_TABLE} SET stage = '{stage}' WHERE stage = '{tmp}'")


def checkpoint(stage, build_fn):
    """Step without LLM: computed once, then read from the cache."""
    if not _forced(stage) and load_stage(stage) is not None:
        print(f"↺ {stage}: cached")
    else:
        replace_stage(stage, build_fn())
        print(f"✓ {stage}: computed")
    return load_stage(stage)


def incremental(stage, df_in, keys, build_fn, ok_col=None):
    """Spark LLM step: only processes the keys missing from the cache (or failed, when ok_col is NULL).
    build_fn(todo) returns a DataFrame with `keys` + the output columns."""
    chunk_level = "chunk_id" in keys
    if _forced(stage):
        _delete_stage(stage)
    done = load_stage(stage, chunk_level)
    cand = df_in.select(*keys).distinct()
    if done is not None:
        ok = done.filter(F.col(ok_col).isNotNull()) if ok_col else done
        cand = cand.join(ok.select(*keys).distinct(), keys, "left_anti")
    todo_keys = cand.collect()          # frozen keys: the step no longer depends on the state of the cache
    print(f"→ {stage}: {len(todo_keys)} row(s) to process")
    if todo_keys:
        kdf = _keys_df(todo_keys, chunk_level)
        todo = df_in.join(kdf, keys, "inner")
        if stage in ESTIMATE_ONLY:
            _DRY[0] = True
            try:
                _estimate(stage, build_fn(todo))
            finally:
                _DRY[0] = False
            print("   Remove the step from ESTIMATE_ONLY to run it.")
            return done
        # Written by parts: an interrupted step keeps what it computed, and the next run resumes from there
        for start in range(0, len(todo_keys), LLM_KEYS_PER_WRITE):
            part = _keys_df(todo_keys[start:start + LLM_KEYS_PER_WRITE], chunk_level)
            _delete_keys(stage, part, chunk_level)
            _append(stage, build_fn(df_in.join(part, keys, "inner")))
            print(f"   {stage}: {min(start + LLM_KEYS_PER_WRITE, len(todo_keys))}/{len(todo_keys)} written")
    return load_stage(stage, chunk_level).join(df_in.select(*keys).distinct(), keys, "inner")


def incremental_py(stage, df_in, compute_fn, schema, ok_col=None):
    """Step computed on the driver (Vector Search, assistant endpoint), keyed by question_id.
    compute_fn(rows) returns tuples in the order of the schema."""
    if _forced(stage):
        _delete_stage(stage)
    done = load_stage(stage)
    todo = df_in
    if done is not None:
        ok = done.filter(F.col(ok_col).isNotNull()) if ok_col else done
        todo = df_in.join(ok.select("question_id").distinct(), "question_id", "left_anti")
    rows = todo.collect()
    print(f"→ {stage}: {len(rows)} case(s) to process")
    if rows:
        out = compute_fn(rows)
        _delete_keys(stage, _keys_df([(r["question_id"],) for r in rows], False))
        if out:
            _append(stage, spark.createDataFrame(out, schema))
    res = load_stage(stage)
    return (res.join(df_in.select("question_id").distinct(), "question_id", "inner")
            if res is not None else spark.createDataFrame([], schema))


# ── Structured LLM calls (JSON schema), sent to the serving endpoint and paced on its rate limits ──
def s_str(desc=None, enum=None):
    d = {"type": "string"}
    if desc:
        d["description"] = desc
    if enum:
        d["enum"] = enum
    return d


def s_int(desc=None):
    return {"type": "integer", **({"description": desc} if desc else {})}


def s_bool(desc=None):
    return {"type": "boolean", **({"description": desc} if desc else {})}


def s_arr(items):
    return {"type": "array", "items": items}


def s_obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def to_ddl(s):
    t = s["type"]
    if t == "string":
        return "STRING"
    if t == "integer":
        return "INT"
    if t == "number":
        return "DOUBLE"
    if t == "boolean":
        return "BOOLEAN"
    if t == "array":
        return f"ARRAY<{to_ddl(s['items'])}>"
    return "STRUCT<" + ", ".join(f"`{k}`: {to_ddl(v)}" for k, v in s["properties"].items()) + ">"


_DRY = [False]


class _Pacer:
    """Keeps the calls of the last minute within LLM_RATE_SHARE of the judge model's token limits."""

    def __init__(self):
        self.lock, self.calls = threading.Lock(), deque()

    def reserve(self, tokens_in, tokens_out) -> list:
        budget_in = LLM_INPUT_TOKENS_PER_MINUTE * LLM_RATE_SHARE
        budget_out = LLM_OUTPUT_TOKENS_PER_MINUTE * LLM_RATE_SHARE
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0][0] > 60:
                    self.calls.popleft()
                used_in = sum(c[1] for c in self.calls)
                used_out = sum(c[2] for c in self.calls)
                if not self.calls or (used_in + tokens_in <= budget_in and used_out + tokens_out <= budget_out):
                    entry = [now, tokens_in, tokens_out]
                    self.calls.append(entry)
                    return entry
            time.sleep(1)

    def settle(self, entry, tokens_in, tokens_out):
        with self.lock:
            entry[1], entry[2] = tokens_in, tokens_out


_PACER = _Pacer()
_TRANSIENT = ("429", "rate limit", "too many requests", "502", "503", "504", "timed out", "timeout", "temporarily")


def _call_llm(model, prompt, response_format) -> tuple:
    """(JSON text, error) of one structured call to the serving endpoint; rate-limited and transient errors are
    retried with a growing wait."""
    body = {"messages": [{"role": "user", "content": prompt}], "response_format": response_format}
    tokens_in, tokens_out = len(prompt) / CHARS_PER_TOKEN, DEFAULT_OUT_CHARS / CHARS_PER_TOKEN
    error = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        entry = _PACER.reserve(tokens_in, tokens_out)
        try:
            resp = w.api_client.do("POST", f"/serving-endpoints/{model}/invocations", body=body)
            used = resp.get("usage") or {}
            _PACER.settle(entry, used.get("prompt_tokens") or tokens_in, used.get("completion_tokens") or tokens_out)
            content = resp["choices"][0]["message"]["content"]
            if isinstance(content, list):          # content parts
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            return content, None
        except Exception as e:
            error = str(e)[:500]
            if not any(t in error.lower() for t in _TRANSIENT):
                break
            time.sleep(min(2 ** (attempt + 1), 60))
    return None, error


def llm(df, prompt_col, out_col, props, model=JUDGE_MODEL):
    """Adds `out_col` (typed struct), `out_col_error`, and prompt/response sizes for cost tracking. Each distinct prompt
    is sent once to the serving endpoint from the driver, LLM_WORKERS at a time, paced on its token limits."""
    schema = s_obj(props)
    response_format = {"type": "json_schema", "json_schema": {"name": out_col, "schema": schema, "strict": True}}
    key, result, error = f"_{out_col}_key", f"_{out_col}_result", f"_{out_col}_errmsg"
    df = df.withColumn(key, F.sha2(F.coalesce(F.col(prompt_col), F.lit("")), 256))
    prompts = {r[0]: r[1] for r in df.select(key, prompt_col).distinct().collect()}
    answers = {}
    if not _DRY[0]:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futures = {pool.submit(_call_llm, model, p, response_format): k for k, p in prompts.items() if p}
            for i, fut in enumerate(as_completed(futures), 1):
                answers[futures[fut]] = fut.result()
                if i % 100 == 0 or i == len(futures):
                    print(f"   {out_col}: {i}/{len(futures)} calls · {time.time() - t0:.0f} s")
    rows = [(k, *(answers.get(k) or (None, None if _DRY[0] else "empty prompt"))) for k in prompts]
    answers_df = spark.createDataFrame(rows, f"{key} STRING, {result} STRING, {error} STRING")
    return (df.join(answers_df, key, "left")
              .withColumn(out_col, F.from_json(F.col(result), to_ddl(schema)))
              .withColumn(f"{out_col}_error", F.col(error))
              .withColumn(f"{out_col}_in_chars", F.length(prompt_col))
              .withColumn(f"{out_col}_out_chars", F.length(F.col(result)))
              .drop(key, result, error))


def usage(*outs):
    return [c for o in outs for c in (f"{o}_in_chars", f"{o}_out_chars")]


def P(*parts):
    """Concatenates text and columns, turning NULLs into empty strings (NULL would void F.concat)."""
    return F.concat(*[F.lit(p) if isinstance(p, str) else F.coalesce(p.cast("string"), F.lit("")) for p in parts])


# ── Cost ──
LLM_STAGES = {"annotations": ["ann"], "reformulations": ["rf"], "chunk_grades": ["g"],
              "gold_candidates": ["gen_a", "gen_b"], "gold_arbitrated": ["arb"], "gold_verified": ["ver"]}


def _cost(in_chars, out_chars):
    t_in = in_chars / CHARS_PER_TOKEN
    t_out = out_chars / CHARS_PER_TOKEN * OUTPUT_OVERHEAD
    dbu = t_in / 1e6 * DBU_PER_M_INPUT + t_out / 1e6 * DBU_PER_M_OUTPUT
    return t_in, t_out, dbu, dbu * USD_PER_DBU


def _estimate(name, res):
    outs = LLM_STAGES.get(name, [])
    a = res.agg(*[F.sum(f"{o}_in_chars").alias(f"i_{o}") for o in outs],
                *[F.count(f"{o}_in_chars").alias(f"n_{o}") for o in outs]).first()
    t = load_stage(name)
    hist = {o: t.agg(F.avg(f"{o}_out_chars")).first()[0] for o in outs
            if t is not None and f"{o}_out_chars" in t.columns}
    total = 0.0
    for o in outs:
        n, i = a[f"n_{o}"] or 0, a[f"i_{o}"] or 0
        t_in, t_out, dbu, usd = _cost(i, n * (hist.get(o) or DEFAULT_OUT_CHARS))
        total += usd
        print(f"   {o}: {n} calls · ~{t_in / 1e6:.2f}M tokens in · ~{t_out / 1e6:.2f}M tokens out · ${usd:.2f}")
    print(f"💰 Estimate for {name}: ${total:.2f} (no LLM called)")


# ── Document references ──
_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|txt)$", re.I)
_LANG = re.compile(r"[-_. ](%s)$" % "|".join(LANG_SUFFIXES), re.I)
_CODE = re.compile(r"^(?=.*\d)(?=(?:.*[A-Z]){2})[A-Z][A-Z0-9_.\-]{2,28}( (%s))?$" % "|".join(LANG_SUFFIXES))
_BOLD = re.compile(r"\*\*([^*\n]{3,40})\*\*")
_REF_IN_URL = re.compile(r"[?&]ref=([A-Za-z0-9_.\-]+)", re.I)


def _strip(s) -> str:
    return _LANG.sub("", _EXT.sub("", str(s).strip().split("/")[-1]))


def base_ref(s) -> str:
    """Document key, insensitive to language suffix, separators, case and zero padding:
    PRLAT538_FR, PRLAT538.FR → PRLAT538; IN_APO_006 and IN_APO_0006 → INAPO6."""
    groups = re.findall(r"[A-Za-z]+|\d+", _strip(s).upper())
    return "".join(str(int(g)) if g.isdigit() else g for g in groups)


def code_like(text) -> set:
    """Document codes cited in a text: **REF** in bold and ?ref= parameters of links."""
    text = str(text or "")
    return {c.strip() for c in _BOLD.findall(text) + _REF_IN_URL.findall(text) if _CODE.match(c.strip().upper())}


def vs_source_table():
    if REF_SOURCE_TABLE:
        return REF_SOURCE_TABLE
    return w.vector_search_indexes.get_index(VS_INDEX).delta_sync_index_spec.source_table


REFS_BY_BASE = {}


def load_ref_catalog():
    """Document key → exact REF values of the index (all language variants)."""
    if not REFS_BY_BASE:
        for r in spark.table(vs_source_table()).select("REF").distinct().collect():
            if r.REF:
                REFS_BY_BASE.setdefault(base_ref(r.REF), set()).add(r.REF)
    return REFS_BY_BASE


def resolve_refs(refs) -> list:
    catalog = load_ref_catalog()
    tokens = refs if isinstance(refs, (list, tuple, set)) else re.split(r"[,;\n]+", str(refs or ""))
    return sorted({real for t in tokens if len(base_ref(t)) >= 3 for real in catalog.get(base_ref(t), set())})


def refs_in_sources_json(raw) -> list:
    """REFs listed by the assistant in sources_json: [{"rank", "title", "url", "n"}, ...]."""
    try:
        items = json.loads(raw) if raw else []
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


def refs_in_response(raw) -> list:
    """Documents cited by the assistant in a raw endpoint response (citations with a title or a ?ref= URL)."""
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
    return sorted(out)


def vs_search(text, k, refs=None):
    """Hybrid search in the document index, optionally restricted to some documents."""
    if not text or not str(text).strip():
        return []
    kwargs = dict(index_name=VS_INDEX, columns=VS_COLUMNS, query_text=str(text)[:2000], num_results=k, query_type="HYBRID")
    if refs:
        kwargs["filters_json"] = json.dumps({"REF": list(refs)})
    try:
        res = w.vector_search_indexes.query_index(**kwargs)
    except Exception as e:
        print(f"⚠️ vector search: {str(e)[:150]}")
        return []
    cols = [c.name for c in res.manifest.columns]
    return [dict(zip(cols, row)) for row in ((res.result.data_array if res.result else None) or [])]


def chunk_key(text) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", str(text).strip()).encode()).hexdigest()


# ── Assistant calls ──
def is_false(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not bool(x)


def ka_messages(question, history, is_self_contained) -> list:
    """Evaluation input: the question, with its history when it only makes sense in the conversation."""
    msgs = []
    if isinstance(history, (list, tuple, np.ndarray)) and len(history) and is_false(is_self_contained):
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    return msgs + [{"role": "user", "content": question}]


def extract_text(resp) -> str:
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


def ask_ka(messages) -> dict:
    """Response of the assistant, with its trace when the endpoint returns it (databricks_options.return_trace)."""
    last_error = None
    for body in ({"input": messages, "databricks_options": {"return_trace": True}}, {"input": messages},
                 {"messages": messages}):
        try:
            return w.api_client.do("POST", f"/serving-endpoints/{KA_ENDPOINT}/invocations", body=body)
        except Exception as e:
            last_error = e
    raise last_error


_SOURCE_HEADER = re.compile(r"\[Source:\s*([^|\]\n]+)")


def ka_retrieval(raw) -> tuple:
    """(document codes, number of passages) retrieved by the assistant: RETRIEVER steps of the trace it returned;
    (None, None) without a trace."""
    trace = (raw.get("databricks_output") or {}).get("trace") if isinstance(raw, dict) else None
    if not isinstance(trace, dict):
        return None, None
    refs, n = [], 0
    for span in (trace.get("data") or {}).get("spans") or []:
        attributes = span.get("attributes") or {}
        if "RETRIEVER" not in str(span.get("span_type") or attributes.get("mlflow.spanType") or "").upper():
            continue
        out = span.get("outputs", attributes.get("mlflow.spanOutputs"))
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                out = []
        for doc in (d for d in (out if isinstance(out, list) else []) if isinstance(d, dict)):
            n += 1
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            header = _SOURCE_HEADER.search(str(doc.get("page_content") or ""))
            in_uri = _REF_IN_URL.search(str(meta.get("doc_uri") or ""))
            ref = meta.get("REF") or (header.group(1).strip() if header else None) or (in_uri.group(1) if in_uri else None)
            if ref:
                refs.append(str(ref))
    return list(dict.fromkeys(refs)), n

# COMMAND ----------

# DBTITLE 1,1. Extraction — question/answer turns, conversation history and user votes
from pyspark.sql.window import Window


def build_qa():
    users = []
    for u in w.users.list(attributes="id,userName,displayName,groups"):
        uid, email = str(u.id), (u.user_name or "").strip().lower()
        groups = {g.display for g in (u.groups or [])}
        if uid in EXCLUDED_USER_IDS or (email != INCLUDE_EMAIL and groups & EXCLUDED_GROUPS):
            continue
        users.append((uid, email))
    df_users = spark.createDataFrame(users, "user_id string, email string")

    msgs = spark.table(f"{SRC}.chat_messages").filter(~F.coalesce(F.col("deleted"), F.lit(False)))
    w_s = Window.partitionBy("session_id").orderBy("created_at", "id")
    w_h = w_s.rowsBetween(-HISTORY_TURNS, -1)
    df = (msgs
          .withColumn("turn_idx", F.row_number().over(w_s))
          .withColumn("next_role", F.lead("role").over(w_s))
          .withColumn("actual_response", F.lead("content").over(w_s))
          .withColumn("answer_id", F.lead("id").over(w_s))
          .withColumn("retrieved_context", F.lead("sources_json").over(w_s))
          .withColumn("answer_status", F.lead("status").over(w_s))
          .withColumn("history", F.collect_list(F.struct(
              F.col("role").alias("role"), F.col("content").alias("content"))).over(w_h))
          .filter((F.col("role") == "user") & (F.col("next_role") == "assistant")))

    fb = (spark.table(f"{SRC}.chat_feedbacks").groupBy("message_id")
          .agg(F.max((F.col("vote") == "down").cast("int")).alias("_down"),
               F.max((F.col("vote") == "up").cast("int")).alias("_up"),
               F.concat_ws(" || ", F.collect_list("comment")).alias("comment"))
          .withColumn("vote", F.when(F.col("_down") == 1, "down").when(F.col("_up") == 1, "up"))
          .drop("_down", "_up"))
    depth = msgs.filter(F.col("role") == "user").groupBy("session_id").agg(F.count("*").alias("session_user_msg_count"))

    df_log = (df.withColumn("user_id", F.col("user_id").cast("string"))
              .join(df_users, "user_id", "inner")                 # excluded users are really excluded
              .join(depth, "session_id", "left")
              .join(fb, df.answer_id == fb.message_id, "left")
              .select(F.col("id").cast("long").alias("question_id"), F.lit("log").alias("source"),
                      "session_id", "user_id", "turn_idx", F.col("content").alias("question"),
                      F.col("created_at").alias("question_created_at"), "actual_response", "answer_id",
                      "retrieved_context", "answer_status", "history", "session_user_msg_count", "vote", "comment")
              .withColumn("history_text", F.array_join(F.transform(
                  "history", lambda m: F.concat(F.upper(m["role"]), F.lit(": "), m["content"])), "\n"))
              .withColumn("has_sources", F.col("retrieved_context").isNotNull()
                          & ~F.trim(F.col("retrieved_context")).isin("", "[]"))
              .filter(F.length(F.trim("question")) >= 8))
    syn = spark.createDataFrame([(int(s["question_id"]), "synthetic", s["question"]) for s in SYNTHETIC_QUERIES],
                                "question_id long, source string, question string")
    return df_log.unionByName(syn, allowMissingColumns=True)


df_qa = checkpoint("qa_pairs", build_qa)
print(f"{df_qa.count()} question/answer turns (logs + synthetic)")

# COMMAND ----------

# DBTITLE 1,2. Annotation — one structured call per question; drives the selection
INTENTS = ["definition_acronym", "document_lookup", "procedure_howto", "rule_requirement", "requirement_compliance",
           "comparison_multi_doc", "link_or_navigation", "chitchat_or_meta", "out_of_scope"]

PROMPT_ANNOTATE = """You curate an evaluation dataset for Qualibot, a RAG assistant answering questions about the QUALITY
documentation of an aerospace manufacturer (procedures, work instructions, forms, templates, quality rules; documents
identified by codes such as PRLAT508, INAQ742, QP-1457, NF-10065).

You receive one real turn from the production logs: the conversation history (may be empty), the question, the answer
produced by Qualibot and the user's feedback, if any.

Fields:
- standalone_question: if the question only makes sense with the history, rewrite it as a self-contained question, in the
  user's language and style, without adding information absent from the history. Otherwise copy it verbatim.
- is_self_contained: true if the original question is understandable without the history.
- intent: definition_acronym, document_lookup (find a document or template), procedure_howto (how to, steps),
  rule_requirement (rule, requirement, threshold, deadline, responsibility), requirement_compliance (whether the company
  complies with a customer or standard requirement, and which document demonstrates it), comparison_multi_doc,
  link_or_navigation, chitchat_or_meta, out_of_scope.
- difficulty: easy (one passage is enough), medium (several passages or documents), hard (ambiguous, expert reasoning,
  scattered information).
- question_quality (1-5): clarity, specificity, realism. 1 = unintelligible or test message.
- eval_value (1-5): value as a test case. 5 = realistic, precise, representative of a real need, with an answer that can be
  checked in the documentation (or clearly out of scope). 1 = no value (greeting, test, trivial duplicate, bare link request).
- ka_answer_assessment: a priori impression of Qualibot's answer, WITHOUT access to the documents: looks_correct, partial,
  wrong_or_hallucinated, refusal, unclear. For a synthetic question without answer: unclear.
- feedback_triage: legitimate (real RAG failure), user_error (out-of-scope expectation, ill-posed question), ambiguous,
  no_feedback (no negative feedback).
- analysis: 1 to 3 sentences justifying the choices."""

ANNOTATE_PROPS = {
    "analysis": s_str(),
    "standalone_question": s_str(),
    "is_self_contained": s_bool(),
    "language": s_str(enum=["fr", "en", "other"]),
    "intent": s_str(enum=INTENTS),
    "topic": s_str("topic in 2 to 5 words"),
    "in_scope": s_str(enum=["yes", "partial", "no"]),
    "difficulty": s_str(enum=["easy", "medium", "hard"]),
    "question_quality": s_int(),
    "eval_value": s_int(),
    "ka_answer_assessment": s_str(enum=["looks_correct", "partial", "wrong_or_hallucinated", "refusal", "unclear"]),
    "feedback_triage": s_str(enum=["legitimate", "user_error", "ambiguous", "no_feedback"]),
}


def prompt_annotations(d):
    return d.withColumn("_p", P(
        PROMPT_ANNOTATE,
        "\n\n### History\n", F.coalesce(F.nullif(F.col("history_text"), F.lit("")), F.lit("(none)")),
        "\n\n### Question\n", F.col("question"),
        "\n\n### Qualibot answer\n", F.coalesce(F.col("actual_response"), F.lit("(none: synthetic question)")),
        "\n\n### Sources returned by Qualibot\n", F.when(F.col("has_sources"), "yes").otherwise("no"),
        "\n\n### User feedback\n", F.concat_ws(" — ", F.coalesce(F.col("vote"), F.lit("none")), F.col("comment"))))


def build_annotations(d):
    return llm(prompt_annotations(d), "_p", "ann", ANNOTATE_PROPS).select("question_id", "ann.*", "ann_error", *usage("ann"))


df_qa = load_stage("qa_pairs")
df_ann = df_qa.join(incremental("annotations", df_qa, ["question_id"], build_annotations, ok_col="intent"), "question_id")
display(df_ann.groupBy("intent", "in_scope").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,3. Shortlist — embeddings, topic clusters, quotas (production failures included)
from sklearn.cluster import KMeans

df_ann = load_stage("qa_pairs").join(load_stage("annotations"), "question_id")


def build_embeddings(d):
    return (d.withColumn("_e", F.expr(f"ai_query('{EMBED_MODEL}', standalone_question, failOnError => false)"))
             .select("question_id", F.col("_e.result").alias("emb"), F.col("_e.errorMessage").alias("emb_error")))


df_emb = incremental("embeddings", df_ann.select("question_id", "standalone_question"), ["question_id"],
                     build_embeddings, ok_col="emb")
pdf = (df_ann.join(df_emb.select("question_id", "emb"), "question_id")
       .filter(F.col("emb").isNotNull() & F.col("intent").isNotNull()).toPandas().reset_index(drop=True))
E = np.vstack(pdf["emb"].apply(lambda v: np.asarray(v, dtype=np.float32)).values)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9

# Turns judged bad (or voted down) by the production quality scoring: the most valuable test cases. The stage at fault
# (retrieval or generation) and unsupported compliance claims come from the same scoring.
try:
    _scores = spark.table(PROD_SCORES_TABLE)
    _extra = [c for c in ("error_source", "compliance_claim") if c in _scores.columns]
    prod_bad = {str(r["message_id"]): r.asDict() for r in _scores.filter("golden_candidate")
                                                             .select("message_id", *_extra).collect()}
    # Turns judged good and not voted down: regression cases (the assistant must keep answering them well)
    prod_good = {str(r["message_id"]) for r in _scores.filter("turn_verdict = 'good' AND NOT golden_candidate")
                                                      .select("message_id").collect()}
except Exception:
    prod_bad, prod_good = {}, set()
_prod = lambda a: prod_bad.get(str(a)) if pd.notna(a) else None
pdf["production_failure"] = pdf["answer_id"].map(lambda a: _prod(a) is not None)
pdf["production_pass"] = pdf["answer_id"].map(lambda a: pd.notna(a) and str(a) in prod_good)
pdf["production_error_source"] = pdf["answer_id"].map(lambda a: (_prod(a) or {}).get("error_source"))
pdf["production_compliance_claim"] = pdf["answer_id"].map(
    lambda a: (_prod(a) or {}).get("compliance_claim") == "unsupported_compliance_claim")
print(f"{len(prod_bad)} production failure(s) available, {int(pdf['production_failure'].sum())} among the annotated turns "
      f"(retrieval at fault: {int(pdf['production_error_source'].isin(['retrieval', 'retrieval_and_generation']).sum())}, "
      f"unsupported compliance claims: {int(pdf['production_compliance_claim'].sum())}) · "
      f"{int(pdf['production_pass'].sum())} annotated turns judged good (regression cases)")

is_log = (pdf["source"] == "log").values
km = KMeans(n_clusters=min(N_CLUSTERS, max(2, is_log.sum() // 5)), n_init=10, random_state=42)
pdf["cluster"] = -1
pdf.loc[is_log, "cluster"] = km.fit_predict(E[is_log])
pdf["cluster_size"] = pdf["cluster"].map(pdf.loc[is_log, "cluster"].value_counts()).fillna(0)


def case_score(r):
    s = float(r.eval_value or 0) + 0.5 * float(r.question_quality or 0)
    if r.vote == "down":
        s += {"legitimate": 2.0, "ambiguous": 1.0}.get(r.feedback_triage, 0.0)
    if r.production_failure:
        s += 1.5
    if r.ka_answer_assessment in ("wrong_or_hallucinated", "partial", "refusal"):
        s += 1.0
    s += {"hard": 1.0, "medium": 0.5}.get(r.difficulty, 0.0)
    if is_false(r.is_self_contained):
        s += 0.5
    if r.answer_status == "error" or is_false(r.has_sources):
        s += 0.5
    return s + 0.3 * math.log1p(r.cluster_size)       # representativeness of the topic


pdf["score"] = pdf.apply(case_score, axis=1)

SLOT_PREDICATES = {
    "production_retrieval_miss": lambda r: r.production_error_source in ("retrieval", "retrieval_and_generation"),
    "production_compliance_claim": lambda r: bool(r.production_compliance_claim),
    "production_failure": lambda r: bool(r.production_failure),
    "production_pass": lambda r: bool(r.production_pass),
    "negative_feedback":  lambda r: r.vote == "down" and r.feedback_triage in ("legitimate", "ambiguous"),
    "suspect_answer":     lambda r: r.ka_answer_assessment in ("wrong_or_hallucinated", "partial", "refusal"),
    "multi_doc_or_hard":  lambda r: r.intent == "comparison_multi_doc" or r.difficulty == "hard",
    "document_lookup":    lambda r: r.intent == "document_lookup",
    "procedure":          lambda r: r.intent == "procedure_howto",
    "requirement_compliance": lambda r: r.intent == "requirement_compliance",
    "rule_requirement":   lambda r: r.intent == "rule_requirement",
    "definition_acronym": lambda r: r.intent == "definition_acronym",
    "multi_turn":         lambda r: is_false(r.is_self_contained),
    "out_of_scope":       lambda r: r.intent == "out_of_scope" or r.in_scope == "no",
    "other_language":     lambda r: r.language != "fr",
}


def eligible(r):
    if r.source != "log" or r.question_id in CURATED_IDS:
        return False
    if r.intent == "out_of_scope" or r.in_scope == "no":
        return (r.question_quality or 0) >= 3
    return r.intent not in EXCLUDED_INTENTS and (r.question_quality or 0) >= 3 and (r.eval_value or 0) >= 3


pos_curated = [i for i in range(len(pdf)) if pdf.at[i, "question_id"] in CURATED_IDS]
missing_curated = CURATED_IDS - set(pdf.loc[pos_curated, "question_id"])
if missing_curated:
    print(f"⚠️ curated questions not found among the annotated turns: {missing_curated}")
candidates = [i for i in range(len(pdf)) if eligible(pdf.iloc[i])]

chosen, slot_of, used_clusters = list(pos_curated), {i: "curated" for i in pos_curated}, Counter()
for slot, quota in SLOT_QUOTAS.items():
    cand = [i for i in candidates if i not in slot_of and SLOT_PREDICATES[slot](pdf.iloc[i])]
    for _ in range(quota):
        if not cand:
            break
        max_sim = (E[cand] @ E[chosen].T).max(axis=1) if chosen else np.zeros(len(cand))
        val = (pdf["score"].values[cand]
               - CLUSTER_PENALTY * np.array([used_clusters[pdf.at[i, "cluster"]] for i in cand])
               - 3.0 * np.clip(max_sim - 0.75, 0, None))
        val[max_sim > DEDUP_SIM] = -np.inf
        j = int(np.argmax(val))
        if not np.isfinite(val[j]):
            break
        i = cand.pop(j)
        chosen.append(i)
        slot_of[i] = slot
        used_clusters[pdf.at[i, "cluster"]] += 1

override_ids = {o["question_id"] for o in MANUAL_OVERRIDES}
sel_rows = [(int(pdf.at[i, "question_id"]),
             "synthetic" if pdf.at[i, "question_id"] < 0 else ("override" if pdf.at[i, "question_id"] in override_ids else "log"),
             slot_of[i], float(pdf.at[i, "score"]), int(pdf.at[i, "cluster"]), bool(pdf.at[i, "production_failure"]),
             pdf.at[i, "production_error_source"] if pd.notna(pdf.at[i, "production_error_source"]) else None)
            for i in chosen]
df_sel = spark.createDataFrame(sel_rows, "question_id long, source string, slot string, score double, cluster int, "
                                         "production_failure boolean, production_error_source string")
df_hints = spark.createDataFrame([(int(o["question_id"]), o.get("expected_response"), o.get("expected_sources"))
                                  for o in MANUAL_OVERRIDES + SYNTHETIC_QUERIES],
                                 "question_id long, hint_response string, hint_sources string")
replace_stage("shortlist", df_sel.join(df_ann.drop("source"), "question_id").join(df_hints, "question_id", "left"))
df_short = load_stage("shortlist")
print(f"Shortlist: {df_short.count()} cases")
display(df_short.groupBy("source", "slot").count().orderBy("source", "slot"))

# COMMAND ----------

# DBTITLE 1,4.1 Current assistant answers — same input as the evaluation (history included for follow-ups)
df_short = load_stage("shortlist")


def compute_ka(rows):
    out = []
    with ThreadPoolExecutor(max_workers=KA_MAX_CONCURRENT) as pool:
        futures = {pool.submit(ask_ka, ka_messages(r.question, r.history, r.is_self_contained)): int(r.question_id)
                   for r in rows}
        for fut in as_completed(futures):
            try:
                raw = fut.result()
                retrieved, n_passages = ka_retrieval(raw)
                raw = {k: v for k, v in raw.items() if k != "databricks_output"} if isinstance(raw, dict) else raw
                out.append((futures[fut], extract_text(raw), refs_in_response(raw), None, retrieved, n_passages))
            except Exception as e:
                out.append((futures[fut], None, [], str(e)[:500], None, None))
    return out


KA_SCHEMA = ("question_id long, ka_fresh_response string, ka_fresh_refs array<string>, ka_fresh_error string, "
             "ka_fresh_retrieved_refs array<string>, ka_fresh_retrieved_count int")
if QUERY_KA_FRESH:
    df_ka = incremental_py("ka_fresh", df_short.select("question_id", "question", "history", "is_self_contained"),
                           compute_ka, KA_SCHEMA, ok_col="ka_fresh_response")
    display(df_ka.groupBy(F.col("ka_fresh_response").isNotNull().alias("answered"),
                          F.col("ka_fresh_retrieved_count").isNotNull().alias("retrieval_known")).count())


def get_ka():
    k = load_stage("ka_fresh")
    if k is None:
        return spark.createDataFrame([], KA_SCHEMA)
    for column, kind in [("ka_fresh_refs", "array<string>"), ("ka_fresh_retrieved_refs", "array<string>"),
                         ("ka_fresh_retrieved_count", "int")]:
        if column not in k.columns:
            k = k.withColumn(column, F.lit(None).cast(kind))
    return k

# COMMAND ----------

# DBTITLE 1,4.2 Search queries — keywords, translation, probable title, hypothetical answer, cited codes
PROMPT_REFORM = """You prepare queries for a hybrid (keyword + semantic) search in an index of aerospace quality documentation
written in French and English (documents are often suffixed _FR / _GB).
- search_queries: exactly 3 queries: (1) a keyword-rich version in the language of the question (expand acronyms, add
  business synonyms); (2) its translation into the other language (French ↔ English); (3) the probable title or section
  heading of the document that contains the answer.
- hypothetical_answer: a short passage (3 to 5 sentences), written as it would appear in an internal quality procedure,
  that would answer the question. It is only used as a search query, so plausible wording matters more than accuracy.
- mentioned_refs: document codes explicitly cited in the question, the history or Qualibot's answers (e.g. PRLAT508,
  INAQ742_FR, QP-1457). Empty list if none."""


def build_reform(d):
    d = d.withColumn("_p", P(PROMPT_REFORM,
                             "\n\n### Question\n", F.col("standalone_question"),
                             "\n\n### History\n", F.col("history_text"),
                             "\n\n### Qualibot answer (logs)\n", F.col("actual_response"),
                             "\n\n### Qualibot answer (current)\n", F.col("ka_fresh_response")))
    props = {"search_queries": s_arr(s_str()), "hypothetical_answer": s_str(), "mentioned_refs": s_arr(s_str())}
    return llm(d, "_p", "rf", props).select("question_id", "rf.search_queries", "rf.hypothetical_answer",
                                            "rf.mentioned_refs", "rf_error", *usage("rf"))


df_reform_in = df_short.join(get_ka().select("question_id", "ka_fresh_response"), "question_id", "left")
df_reform = incremental("reformulations", df_reform_in, ["question_id"], build_reform, ok_col="search_queries")

# COMMAND ----------

# DBTITLE 1,5. Evidence pool — every retrieval route, merged and de-duplicated
df_short = load_stage("shortlist")
load_ref_catalog()
print(f"{len(REFS_BY_BASE)} documents in the index")
_ref_warnings = Counter()


def _pool_one(r) -> list:
    chunks = {}

    def add(hit, origin):
        text = hit.get("chunk_text") or ""
        if not text.strip():
            return
        cid = chunk_key(text)
        c = chunks.setdefault(cid, {"REF": hit.get("REF"), "semantic_headers": hit.get("semantic_headers"),
                                    "chunk_text": text, "origins": set(), "best_score": 0.0})
        c["origins"].add(origin)
        c["best_score"] = max(c["best_score"], float(hit.get("score") or 0.0))

    q, sq = r.question, (r.standalone_question or r.question)
    queries = [("question", q, 10)]
    if sq.strip() != q.strip():
        queries.append(("standalone", sq, 10))
    queries += [("reformulation", x, 8) for x in (r.search_queries or [])]
    if getattr(r, "hypothetical_answer", None):
        queries.append(("hypothetical_answer", r.hypothetical_answer, 8))
    for origin, text, k in queries:
        for hit in vs_search(text, k):
            add(hit, origin)

    groups = {
        "hint": resolve_refs(r.hint_sources),
        "assistant_logs": resolve_refs(refs_in_sources_json(r.retrieved_context)),
        "assistant_current": resolve_refs(list(r.ka_fresh_refs or [])),
        "assistant_retrieval": resolve_refs(list(r.ka_fresh_retrieved_refs or [])),
        "mentioned": resolve_refs(list(r.mentioned_refs or []) + sorted(code_like(r.actual_response))
                                  + sorted(code_like(r.ka_fresh_response))),
    }
    for origin, refs in groups.items():
        if refs:
            for hit in vs_search(sq, min(24, 6 * len(refs)), refs=refs):
                add(hit, origin)
    cited = refs_in_sources_json(r.retrieved_context) + list(r.hint_sources.split(",") if r.hint_sources else [])
    for c in cited:
        if not resolve_refs([c]):
            _ref_warnings[c.strip()] += 1

    priority = {"hint", "assistant_logs", "assistant_current", "assistant_retrieval", "mentioned"}
    items = sorted(chunks.items(), key=lambda kv: (bool(kv[1]["origins"] & priority), len(kv[1]["origins"]),
                                                   kv[1]["best_score"]), reverse=True)[:MAX_POOL_CHUNKS]
    return [(int(r.question_id), cid, c["REF"], c["semantic_headers"], c["chunk_text"], sorted(c["origins"]),
             float(c["best_score"])) for cid, c in items]


def compute_pool(rows):
    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(_pool_one, r) for r in rows]):
            out.extend(fut.result())
    if _ref_warnings:
        print(f"⚠️ codes not found in the index: {dict(_ref_warnings)}")
    return out


POOL_SCHEMA = ("question_id long, chunk_id string, REF string, semantic_headers string, chunk_text string, "
               "origins array<string>, best_score double")
df_pool_in = (df_short
              .join(load_stage("reformulations").select("question_id", "search_queries", "hypothetical_answer", "mentioned_refs")
                    if "hypothetical_answer" in load_stage("reformulations").columns
                    else load_stage("reformulations").select("question_id", "search_queries", "mentioned_refs")
                         .withColumn("hypothetical_answer", F.lit(None).cast("string")), "question_id", "left")
              .join(get_ka().select("question_id", "ka_fresh_response", "ka_fresh_refs", "ka_fresh_retrieved_refs"),
                    "question_id", "left"))
df_pool = incremental_py("evidence_pool", df_pool_in, compute_pool, POOL_SCHEMA)
display(df_pool.groupBy("question_id").agg(F.count("*").alias("chunks"), F.countDistinct("REF").alias("documents")).summary())

# COMMAND ----------

# DBTITLE 1,6. Chunk grading, expansion around the best chunks (similar and adjacent chunks), second grading pass
PROMPT_GRADE = """You are a strict relevance assessor for a RAG benchmark on aerospace quality documentation.
Judge whether the EXCERPT helps answer the QUESTION.
- 3: contains information that directly answers the question (fully or an essential part of it).
- 2: contains a complement a complete answer should mention (condition, exception, related step, identity of the
     reference document).
- 1: same general topic, but does not help answer.
- 0: unrelated.
An approximate human hint of the expected answer may be given: use it to recognise the information sought, but judge only
what the excerpt contains.
extracted_facts: when relevance >= 2, the facts of the excerpt useful for the answer, self-contained and faithful to the
text, in the excerpt's language; otherwise an empty list. analysis: one sentence."""


def build_grades(d):
    d = d.withColumn("_p", P(PROMPT_GRADE,
                             "\n\n### Question\n", F.col("standalone_question"),
                             "\n\n### Human hint (may be empty)\n", F.col("hint_response"),
                             "\n\n### Excerpt [", F.col("REF"), " | ", F.col("semantic_headers"), "]\n", F.col("chunk_text")))
    props = {"analysis": s_str(), "relevance": s_int(), "extracted_facts": s_arr(s_str())}
    return llm(d, "_p", "g", props).select("question_id", "chunk_id", "g.relevance", "g.extracted_facts",
                                           F.col("g.analysis").alias("grade_analysis"), "g_error", *usage("g"))


def all_evidence():
    """First-pass pool ∪ expansion chunks (similar chunks of the same document, adjacent chunks)."""
    out = load_stage("evidence_pool")
    for stage in ("evidence_expansion", "evidence_neighbours"):
        extra = load_stage(stage)
        if extra is not None:
            out = out.unionByName(extra)
    return out


def grade(evidence):
    grade_in = evidence.join(df_short.select("question_id", "standalone_question", "hint_response"), "question_id")
    return incremental("chunk_grades", grade_in, ["question_id", "chunk_id"], build_grades, ok_col="relevance")


df_short = load_stage("shortlist")
grade(load_stage("evidence_pool"))


# Expansion: other chunks of the documents that hold a directly relevant chunk (continuations, tables, exceptions)
def _expand_one(r) -> list:
    known, out = set(r.known_ids), {}
    for c in r.best_chunks:
        for hit in vs_search(c["chunk_text"][:800], EXPANSION_PER_CHUNK + 1, refs=[c["REF"]]):
            cid = chunk_key(hit.get("chunk_text") or "")
            if hit.get("chunk_text") and cid not in known and cid not in out:
                out[cid] = (int(r.question_id), cid, hit.get("REF"), hit.get("semantic_headers"), hit["chunk_text"],
                            ["expansion"], float(hit.get("score") or 0.0))
    return list(out.values())


def compute_expansion(rows):
    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(_expand_one, r) for r in rows]):
            out.extend(fut.result())
    return out


graded = load_stage("chunk_grades").select("question_id", "chunk_id", "relevance")
pool = load_stage("evidence_pool")
exp_in = (pool.join(graded, ["question_id", "chunk_id"])
          .groupBy("question_id")
          .agg(F.collect_list(F.when(F.col("relevance") == 3, F.struct("REF", "chunk_text"))).alias("best_chunks"),
               F.collect_list("chunk_id").alias("known_ids"))
          .filter(F.size("best_chunks") > 0))
incremental_py("evidence_expansion", exp_in, compute_expansion, POOL_SCHEMA)


# Neighbours: the chunks just before and after each directly relevant chunk, in document order (chunk_index of the
# index source table). They hold continuations of procedures, table rows and exceptions that similarity misses.
def compute_neighbours(rows):
    """rows: one per case with best_chunks = [(REF, chunk_text)] graded 3 and known_ids = chunks already pooled."""
    best = spark.createDataFrame([(int(r.question_id), c["REF"], c["chunk_text"]) for r in rows for c in r.best_chunks],
                                 "question_id long, REF string, chunk_text string")
    src = spark.table(vs_source_table()).select("IDDOC", "REF", "chunk_index", "chunk_text", "semantic_headers")
    anchors = best.join(src.select("IDDOC", "REF", "chunk_index", "chunk_text"), ["REF", "chunk_text"]) \
                  .select("question_id", "IDDOC", F.col("chunk_index").alias("anchor_index"))
    hits = (anchors.join(src, "IDDOC")
            .filter((F.abs(F.col("chunk_index") - F.col("anchor_index")) <= NEIGHBOUR_WINDOW)
                    & (F.col("chunk_index") != F.col("anchor_index")))
            .select("question_id", "REF", "semantic_headers", "chunk_text").collect())
    known = {int(r.question_id): set(r.known_ids) for r in rows}
    out = {}
    for h in hits:
        cid = chunk_key(h.chunk_text or "")
        if h.chunk_text and cid not in known[int(h.question_id)]:
            out[(int(h.question_id), cid)] = (int(h.question_id), cid, h.REF, str(h.semantic_headers or ""),
                                              h.chunk_text, ["neighbour"], 0.0)
    return list(out.values())


pooled = all_evidence().select("question_id", "chunk_id")
nb_in = (all_evidence().join(load_stage("chunk_grades").select("question_id", "chunk_id", "relevance"), ["question_id", "chunk_id"])
         .groupBy("question_id")
         .agg(F.collect_list(F.when(F.col("relevance") == 3, F.struct("REF", "chunk_text"))).alias("best_chunks"))
         .join(pooled.groupBy("question_id").agg(F.collect_list("chunk_id").alias("known_ids")), "question_id")
         .filter(F.size("best_chunks") > 0))
incremental_py("evidence_neighbours", nb_in, compute_neighbours, POOL_SCHEMA)

grades = grade(all_evidence())
display(grades.groupBy("relevance").count().orderBy("relevance"))

# COMMAND ----------

# DBTITLE 1,6.1 Evidence per case, and retrieval diagnostics
def evidence_df():
    j = (all_evidence().join(load_stage("chunk_grades").select("question_id", "chunk_id", "relevance"), ["question_id", "chunk_id"])
         .withColumn("_txt", P("[REF: ", F.col("REF"), " | ", F.substring("semantic_headers", 1, 150),
                               " | relevance ", F.col("relevance"), "]\n", F.col("chunk_text"))))
    agg = (j.filter(F.col("relevance") >= 2).groupBy("question_id").agg(
        F.sort_array(F.collect_list(F.struct("relevance", "best_score", "_txt")), asc=False).alias("_top"),
        F.array_distinct(F.collect_list(F.when(F.col("relevance") == 3, F.col("REF")))).alias("refs_rel3"),
        F.array_distinct(F.collect_list(F.when(F.col("relevance") == 2, F.col("REF")))).alias("refs_rel2"),
        F.max(((F.col("relevance") == 3) & F.array_contains("origins", "question")).cast("int")).alias("found_by_question_query"),
        F.max(((F.col("relevance") == 3) & (F.array_contains("origins", "assistant_logs")
                                            | F.array_contains("origins", "assistant_current")
                                            | F.array_contains("origins", "assistant_retrieval"))).cast("int")).alias("assistant_retrieved_relevant"),
        F.max(((F.col("relevance") == 3) & F.array_contains("origins", "expansion")).cast("int")).alias("found_by_expansion"),
        F.max(((F.col("relevance") == 3) & F.array_contains("origins", "neighbour")).cast("int")).alias("found_by_neighbour"),
        F.count("*").alias("n_relevant_chunks"))
        .withColumn("context", F.concat_ws("\n---\n", F.transform("_top", lambda e: e["_txt"])))
        .withColumn("context_rev", F.concat_ws("\n---\n", F.transform(F.reverse("_top"), lambda e: e["_txt"])))
        .drop("_top"))
    pool_n = all_evidence().groupBy("question_id").agg(F.count("*").alias("n_pool_chunks"))
    return (load_stage("shortlist").select("question_id").join(pool_n, "question_id", "left").join(agg, "question_id", "left")
            .withColumn("context", F.coalesce("context", F.lit("(no relevant excerpt found)")))
            .withColumn("context_rev", F.coalesce("context_rev", F.lit("(no relevant excerpt found)"))))


display(evidence_df().drop("context", "context_rev"))

# COMMAND ----------

# DBTITLE 1,7. Reference answers — two independent generations (A/B, excerpts in reverse order for B)
PROMPT_GEN = """You write the REFERENCE ANSWER of an evaluation dataset for Qualibot, a RAG assistant on the quality
documentation of an aerospace manufacturer.

Absolute rules:
- Use ONLY the excerpts provided. No outside knowledge, no assumption, no invented link.
- An approximate human hint may be given: it points to the right document or idea, but the excerpts prevail. If it
  contradicts the excerpts, follow the excerpts and say so in missing_information.

Fields:
- answerability: full (the excerpts fully answer), partial (incomplete answer), none (the excerpts do not allow an answer),
  out_of_scope (unrelated to the quality documentation).
- expected_facts: 1 to 8 atomic facts a good answer must contain (one fact = one checkable statement: value, deadline,
  responsible role, step, document code…), each tied to the REF of the excerpt that proves it. Empty for none/out_of_scope.
- expected_response: the ideal answer as Qualibot should give it — same language as the question, markdown, documents
  cited in the text (« According to **PRLAT508**, … »), concise and actionable. If information is missing, say what is
  missing and give what is partially available. Out of scope: politely explain that Qualibot only covers the quality
  documentation.
- expected_sources: REF codes of the documents that are indispensable to the answer.
- missing_information: what is missing or ambiguous (empty if nothing).

Compliance questions (does the company meet a customer or standard requirement, which document proves it): the expected
answer names the internal documents that address the requirement and states what they establish; it concludes on
compliance only as far as they establish it, and says explicitly what the excerpts do not demonstrate."""

GEN_PROPS = {
    "analysis": s_str(),
    "answerability": s_str(enum=["full", "partial", "none", "out_of_scope"]),
    "expected_facts": s_arr(s_obj({"fact": s_str(), "source_ref": s_str()})),
    "expected_response": s_str(),
    "expected_sources": s_arr(s_str()),
    "missing_information": s_str(),
}


def _gen_prompt(ctx_col):
    return P(PROMPT_GEN,
             "\n\n### Question\n", F.col("standalone_question"),
             "\n\n### Human hint (may be empty)\n", F.col("hint_response"),
             "\n\n### Documents indicated by a human (may be empty)\n", F.col("hint_sources"),
             "\n\n### Excerpts (sorted by relevance)\n", F.col(ctx_col))


def build_gen(d):
    d = d.withColumn("_pa", _gen_prompt("context")).withColumn("_pb", _gen_prompt("context_rev"))
    d = llm(llm(d, "_pa", "gen_a", GEN_PROPS), "_pb", "gen_b", GEN_PROPS)
    return d.select("question_id", "gen_a", "gen_b", "gen_a_error", "gen_b_error", *usage("gen_a", "gen_b"))


df_gen_in = load_stage("shortlist").join(evidence_df(), "question_id", "left")
df_gen = incremental("gold_candidates", df_gen_in, ["question_id"], build_gen, ok_col="gen_a")

# COMMAND ----------

# DBTITLE 1,8. Arbitration — merge A/B, confront the assistant's answers, verdicts, guidelines, confidence
PROMPT_ARB = """You are the final arbiter of an evaluation dataset for Qualibot (RAG on aerospace quality documentation).
You have:
- the relevant EXCERPTS: the ONLY source of truth;
- two reference candidates A and B, generated independently;
- Qualibot's answer in production (logs) and its current answer;
- the user's feedback and an optional human hint.

Build the final reference:
- expected_facts: merge the correct facts of A and B. Add a fact given by Qualibot only if an excerpt proves it. Remove any
  fact the excerpts do not prove. importance = essential when a correct answer MUST contain it (ideally 1 to 5 essential
  facts), secondary otherwise.
- expected_response: the best answer (Qualibot style: language of the question, markdown, **REF** cited in the text,
  concise), containing only proven information.
- expected_sources: indispensable REF codes.
- answerability: full, partial, none or out_of_scope.
- guidelines: 1 to 3 behavioural guidelines a judge can check, only when they matter for this case (e.g. « Must cite
  PRLAT508 », « Must say the information is not in the documentation instead of inventing it »). For a compliance
  question, the reference concludes on compliance only as far as the excerpts establish it.
- ka_log_verdict and ka_fresh_verdict: correct, partially_correct, incorrect, justified_refusal, unjustified_refusal,
  not_available (no answer).
- disagreements: A/B disagreements, contradictions between the hint and the excerpts, or between Qualibot and the excerpts.
- confidence: high (explicit excerpts, A and B agree), medium, low (indirect excerpts, contradictions, ambiguous question);
  confidence_reasons in one or two sentences."""

VERDICTS = ["correct", "partially_correct", "incorrect", "justified_refusal", "unjustified_refusal", "not_available"]
ARB_PROPS = {
    "analysis": s_str(),
    "answerability": s_str(enum=["full", "partial", "none", "out_of_scope"]),
    "expected_facts": s_arr(s_obj({"fact": s_str(), "source_ref": s_str(),
                                   "importance": s_str(enum=["essential", "secondary"])})),
    "expected_response": s_str(),
    "expected_sources": s_arr(s_str()),
    "guidelines": s_arr(s_str()),
    "ka_log_verdict": s_str(enum=VERDICTS),
    "ka_fresh_verdict": s_str(enum=VERDICTS),
    "disagreements": s_str(),
    "confidence": s_str(enum=["high", "medium", "low"]),
    "confidence_reasons": s_str(),
}


def build_arb(d):
    d = d.withColumn("_p", P(
        PROMPT_ARB,
        "\n\n### Question\n", F.col("standalone_question"),
        "\n\n### Human hint\n", F.col("hint_response"), " | documents: ", F.col("hint_sources"),
        "\n\n### User feedback\n", F.concat_ws(" — ", F.col("vote"), F.col("comment")),
        "\n\n### Excerpts\n", F.col("context"),
        "\n\n### Reference A\n", F.to_json("gen_a"),
        "\n\n### Reference B\n", F.to_json("gen_b"),
        "\n\n### Qualibot answer (logs)\n", F.coalesce(F.col("actual_response"), F.lit("(none)")),
        "\n\n### Qualibot answer (current)\n", F.coalesce(F.col("ka_fresh_response"), F.lit("(none)"))))
    return llm(d, "_p", "arb", ARB_PROPS).select("question_id", "arb", "arb_error", *usage("arb"))


df_arb_in = (load_stage("shortlist").join(evidence_df(), "question_id", "left")
             .join(load_stage("gold_candidates"), "question_id")
             .join(get_ka().select("question_id", "ka_fresh_response"), "question_id", "left"))
df_arb = incremental("gold_arbitrated", df_arb_in, ["question_id"], build_arb, ok_col="arb")

# COMMAND ----------

# DBTITLE 1,9. Independent verification — every fact and the reference answer, against the excerpts only
PROMPT_VERIFY = """You are an independent verifier. You receive documentation EXCERPTS, a QUESTION, a numbered list of FACTS
and a reference ANSWER.
- fact_checks: for each fact (by number), supported = true only if an excerpt states it explicitly (rewording accepted,
  extrapolation refused); evidence_ref = REF of the proving excerpt (empty otherwise).
- response_unsupported_claims: statements of the ANSWER absent from the excerpts (courtesy formulas and « information not
  found » statements excluded).
- response_faithful: true if the answer contains no unsupported statement.
- answers_question: true if the answer addresses the question (or correctly explains why it cannot)."""

VERIFY_PROPS = {
    "analysis": s_str(),
    "fact_checks": s_arr(s_obj({"fact_index": s_int(), "supported": s_bool(), "evidence_ref": s_str()})),
    "response_unsupported_claims": s_arr(s_str()),
    "response_faithful": s_bool(),
    "answers_question": s_bool(),
}


def build_verify(d):
    facts_txt = F.array_join(F.transform(
        F.col("arb.expected_facts"), lambda f, i: F.concat(F.lit("["), (i + 1).cast("string"), F.lit("] "), f["fact"])), "\n")
    d = d.withColumn("_p", P(PROMPT_VERIFY,
                             "\n\n### Question\n", F.col("standalone_question"),
                             "\n\n### Excerpts\n", F.col("context"),
                             "\n\n### Facts\n", facts_txt,
                             "\n\n### Answer\n", F.col("arb.expected_response")))
    return llm(d, "_p", "ver", VERIFY_PROPS).select("question_id", "ver", "ver_error", *usage("ver"))


df_ver_in = load_stage("shortlist").join(evidence_df(), "question_id", "left").join(load_stage("gold_arbitrated"), "question_id")
df_ver = incremental("gold_verified", df_ver_in, ["question_id"], build_verify, ok_col="ver")

# COMMAND ----------

# DBTITLE 1,10. Consolidation and final selection
# Final confidence = arbiter confidence (high 3 / medium 2 / low 1), minus 1 for each warning sign: an essential fact
# not verified, an unfaithful reference answer, or generations A and B disagreeing on answerability.
KA_FAIL = {"incorrect", "partially_correct", "unjustified_refusal"}
CONF = {"high": 3, "medium": 2, "low": 1}

pdf_all = (load_stage("shortlist")
           .join(evidence_df().drop("context_rev"), "question_id", "left")
           .join(load_stage("gold_candidates"), "question_id", "left")
           .join(load_stage("gold_arbitrated"), "question_id", "left")
           .join(load_stage("gold_verified"), "question_id", "left")
           .join(get_ka().select("question_id", "ka_fresh_response", "ka_fresh_retrieved_refs", "ka_fresh_retrieved_count"),
                 "question_id", "left")
           .toPandas())


def ka_failure_stage(verdict, expected_sources, retrieved, n_passages):
    """Stage at fault when the current assistant fails the case: retrieval (none of the expected documents among the
    passages it retrieved), generation (it retrieved at least one of them, or no document is expected), unknown (its
    trace was not returned); None when it does not fail."""
    if verdict not in KA_FAIL:
        return None
    if n_passages is None or pd.isna(n_passages):
        return "unknown"
    expected = {base_ref(re.sub(r"^\s*REF\s*:\s*", "", str(d))) for d in expected_sources}
    got = {base_ref(d) for d in (retrieved if retrieved is not None else [])}
    return "generation" if not expected or expected & got else "retrieval"


def _g(obj, attr, default=None):
    if obj is None:
        return default
    try:
        v = obj[attr]
    except (KeyError, ValueError, TypeError):
        return default
    return default if v is None else v


def consolidate(r):
    arb, ver = r.arb, r.ver
    checks = {_g(c, "fact_index"): (_g(c, "supported", False), _g(c, "evidence_ref", "")) for c in _g(ver, "fact_checks", [])}
    essential, secondary, rejected = [], [], []
    for i, f in enumerate(_g(arb, "expected_facts", []), start=1):
        ok, ev = checks.get(i, (False, ""))
        item = {"fact": _g(f, "fact", ""), "source_ref": _g(f, "source_ref", "") or ev}
        (rejected if not ok else essential if _g(f, "importance") == "essential" else secondary).append(item)
    cur = CURATED.get(int(r.question_id), {})
    human = [{"fact": f, "source_ref": "human"} for f in cur.get("human_facts", [])]
    essential = human + [f for f in essential if not any(f["fact"].strip().lower() == h["fact"].strip().lower() for h in human)]
    n_essential_proposed = sum(_g(f, "importance") == "essential" for f in _g(arb, "expected_facts", []))
    conf = CONF.get(_g(arb, "confidence"), 1)
    if n_essential_proposed and len(essential) - len(human) < n_essential_proposed:
        conf -= 1
    if not _g(ver, "response_faithful", False) or not _g(ver, "answers_question", False):
        conf -= 1
    if _g(r.gen_a, "answerability") != _g(r.gen_b, "answerability"):
        conf -= 1
    fresh = _g(arb, "ka_fresh_verdict", "not_available")
    retrieved = list(r.ka_fresh_retrieved_refs) if isinstance(r.ka_fresh_retrieved_refs, (list, np.ndarray)) else None
    notes = [f"Confidence: {_g(arb, 'confidence_reasons', '')}",
             f"Assistant retrieved: {', '.join(retrieved) if retrieved else ''}", f"Disagreements: {_g(arb, 'disagreements', '')}",
             f"Missing (A): {_g(r.gen_a, 'missing_information', '')}",
             f"Unsupported in the reference answer: {'; '.join(_g(ver, 'response_unsupported_claims', []))}",
             f"Rejected facts: {'; '.join(f['fact'] for f in rejected)}", f"Annotation: {r.analysis or ''}"]
    return pd.Series({
        "final_answerability": _g(arb, "answerability"),
        "essential_facts": essential, "secondary_facts": secondary, "rejected_facts": rejected,
        "expected_response": _g(arb, "expected_response", ""),
        "expected_sources": list(_g(arb, "expected_sources", [])),
        "guidelines": list(dict.fromkeys(cur.get("human_guidelines", []) + list(_g(arb, "guidelines", [])))),
        "ka_log_verdict": _g(arb, "ka_log_verdict"), "ka_fresh_verdict": fresh,
        "ka_verdict": fresh if fresh != "not_available" else _g(arb, "ka_log_verdict", "not_available"),
        "ka_failure_stage": ka_failure_stage(fresh, list(_g(arb, "expected_sources", [])), retrieved,
                                             r.ka_fresh_retrieved_count),
        "confidence_arbiter": _g(arb, "confidence"), "confidence_final": max(conf, 0),
        "notes": " | ".join(x for x in notes if not x.rstrip().endswith(":")),
    })


pdf_all = pd.concat([pdf_all, pdf_all.apply(consolidate, axis=1)], axis=1)


def select_final(pdf):
    pdf = pdf[pdf["arb"].notna()].copy()
    pdf["value"] = pdf["score"].fillna(0) + 1.5 * pdf["confidence_final"] + np.where(pdf["ka_verdict"].isin(KA_FAIL), 1.0, 0.0)
    by_id = pdf.set_index("question_id")
    chosen = list(pdf.loc[pdf["source"].isin(["override", "synthetic"]), "question_id"])
    count = lambda pred: sum(pred(by_id.loc[q]) for q in chosen)
    intents = Counter(by_id.loc[chosen, "intent"])

    def fill(cands, n):
        k = 0
        for _, r in cands.sort_values("value", ascending=False).iterrows():
            if k >= n or len(chosen) >= TARGET_N:
                break
            if r.question_id in chosen or intents[r.intent] >= MAX_PER_INTENT:
                continue
            chosen.append(r.question_id)
            intents[r.intent] += 1
            k += 1

    ok = pdf[pdf["confidence_final"] >= 2]
    fill(ok[ok["ka_verdict"].isin(KA_FAIL)], MIN_KA_FAIL - count(lambda r: r.ka_verdict in KA_FAIL))
    fill(ok[ok["intent"] == "requirement_compliance"],
         MIN_COMPLIANCE - count(lambda r: r.intent == "requirement_compliance"))
    fill(ok[ok["final_answerability"].isin(["none", "out_of_scope"])],
         MIN_REFUSAL_CASES - count(lambda r: r.final_answerability in ("none", "out_of_scope")))
    fill(ok[ok["ka_verdict"] == "correct"], MIN_KA_OK - count(lambda r: r.ka_verdict == "correct"))
    fill(pdf[(pdf["confidence_final"] < 2) & pdf["slot"].isin(["negative_feedback", "production_failure",
                                                                "production_retrieval_miss", "production_compliance_claim"])],
         MAX_NEEDS_EXPERT)
    fill(ok, TARGET_N)
    for label, n, minimum in [("cases the assistant fails", count(lambda r: r.ka_verdict in KA_FAIL), MIN_KA_FAIL),
                              ("cases the assistant passes", count(lambda r: r.ka_verdict == "correct"), MIN_KA_OK),
                              ("compliance questions", count(lambda r: r.intent == "requirement_compliance"), MIN_COMPLIANCE),
                              ("refusal cases", count(lambda r: r.final_answerability in ("none", "out_of_scope")),
                               MIN_REFUSAL_CASES)]:
        if n < minimum:
            print(f"⚠️ {label}: {n} selected, {minimum} wanted — not enough such cases in the shortlist")
    out = pdf[pdf["question_id"].isin(chosen)].copy()
    out["needs_expert"] = out["confidence_final"] < 2
    return out.sort_values(["needs_expert", "confidence_final"], ascending=[False, True])


pdf_final = select_final(pdf_all)

GOLDEN_SCHEMA = T.StructType([T.StructField(n, t) for n, t in [
    ("question_id", T.LongType()), ("source", T.StringType()), ("slot", T.StringType()), ("intent", T.StringType()),
    ("difficulty", T.StringType()), ("language", T.StringType()), ("question", T.StringType()),
    ("standalone_question", T.StringType()), ("is_self_contained", T.BooleanType()), ("history_text", T.StringType()),
    ("actual_response", T.StringType()), ("ka_fresh_response", T.StringType()), ("vote", T.StringType()),
    ("comment", T.StringType()), ("production_failure", T.BooleanType()), ("final_answerability", T.StringType()),
    ("expected_response", T.StringType()), ("ka_log_verdict", T.StringType()), ("ka_fresh_verdict", T.StringType()),
    ("ka_verdict", T.StringType()), ("confidence_arbiter", T.StringType()), ("confidence_final", T.LongType()),
    ("needs_expert", T.BooleanType()), ("found_by_question_query", T.DoubleType()),
    ("assistant_retrieved_relevant", T.DoubleType()), ("found_by_expansion", T.DoubleType()),
    ("found_by_neighbour", T.DoubleType()),
    ("n_pool_chunks", T.DoubleType()), ("n_relevant_chunks", T.DoubleType()), ("notes", T.StringType()),
    ("context", T.StringType()), ("essential_facts_json", T.StringType()), ("secondary_facts_json", T.StringType()),
    ("rejected_facts_json", T.StringType()), ("expected_sources_json", T.StringType()), ("guidelines_json", T.StringType()),
    ("messages_json", T.StringType()), ("production_error_source", T.StringType()),
    ("ka_failure_stage", T.StringType())]])


def _py(v, t):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        v = v.item()
    if isinstance(t, T.BooleanType):
        return bool(v)
    if isinstance(t, T.LongType):
        return int(v)
    if isinstance(t, T.DoubleType):
        return float(v)
    return str(v)


save = pdf_final.copy()
for c in ["essential_facts", "secondary_facts", "rejected_facts", "expected_sources", "guidelines"]:
    save[c + "_json"] = pdf_final[c].apply(lambda v: json.dumps(list(v), ensure_ascii=False))
save["messages_json"] = pdf_final.apply(
    lambda r: json.dumps(ka_messages(r["question"], r["history"], r["is_self_contained"]), ensure_ascii=False, default=str), axis=1)
rows = [tuple(_py(r.get(f.name), f.dataType) for f in GOLDEN_SCHEMA.fields) for r in save.to_dict("records")]
replace_stage("golden_final", spark.createDataFrame(rows, GOLDEN_SCHEMA))

print(f"Final selection: {len(pdf_final)} cases")
print(pdf_final.groupby("source").size().to_string())
print(pdf_final["ka_verdict"].value_counts().to_string())
print(pdf_final["final_answerability"].value_counts().to_string())

# COMMAND ----------

# DBTITLE 1,11. Human review — validate / reject / send to an expert (decisions saved in the cache)
import ipywidgets as widgets
from IPython.display import display as ipy_display, HTML   # aliased: keeps Databricks' display() intact

STATUS = ["to_review", "validated", "rejected", "expert"]
_STATUS_ALIASES = {"à revoir": "to_review", "valider": "validated", "rejeter": "rejected"}   # labels of earlier decisions
PAGE_SIZE = 3
MAX_CHARS = {"response": 2500, "context": 5000, "history": 1500, "notes": 1500}
BADGE = {"full": "#28a745", "partial": "#ffc107", "none": "#6c757d", "out_of_scope": "#343a40",
         "correct": "#28a745", "partially_correct": "#fd7e14", "incorrect": "#dc3545",
         "justified_refusal": "#17a2b8", "unjustified_refusal": "#dc3545", "not_available": "#adb5bd"}
_JSON_COLS = ["essential_facts", "secondary_facts", "rejected_facts", "expected_sources", "guidelines"]


def load_golden() -> pd.DataFrame:
    recs = stage_payloads("golden_final")
    if not recs:
        raise ValueError("No final selection in the cache: run section 10 first.")
    pdf = pd.DataFrame(recs)
    for f in GOLDEN_SCHEMA.fields:           # to_json drops NULL fields: restore missing columns
        if f.name not in pdf.columns:
            pdf[f.name] = None
    for c in _JSON_COLS:
        pdf[c] = pdf[c + "_json"].apply(lambda v: json.loads(v) if isinstance(v, str) and v else [])
    pdf["confidence_final"] = pd.to_numeric(pdf["confidence_final"], errors="coerce").fillna(0).astype(int)
    pdf["needs_expert"] = pdf["needs_expert"].fillna(False).astype(bool)
    return pdf.sort_values(["needs_expert", "confidence_final"], ascending=[False, True]).reset_index(drop=True)


def load_decisions() -> dict:
    return {r["question_id"]: (_STATUS_ALIASES.get(r.get("status"), r.get("status") or "to_review"), r.get("note") or "")
            for r in stage_payloads("review")}


def save_decisions(dec: dict):
    rows = [(int(q), s, n) for q, (s, n) in dec.items()]
    if rows:
        replace_stage("review", spark.createDataFrame(rows, "question_id long, status string, note string"))


def _v(r, c):
    v = r.get(c)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _e(x):
    return html_mod.escape("" if x is None else str(x)).replace("\n", "<br>")


def _clip(x, n):
    t = "" if x is None else str(x)
    return _e(t) if len(t) <= n else _e(t[:n]) + f"<br><i style='color:#999'>… truncated ({len(t)} characters)</i>"


def _badge(text, label):
    if not text:
        return ""
    return (f"<span style='font-size:.65em;color:#999;margin-right:2px'>{label}</span><span style='background:"
            f"{BADGE.get(str(text), '#6c757d')};color:#fff;padding:2px 8px;border-radius:10px;font-size:.72em;"
            f"font-weight:600;margin-right:8px'>{_e(text)}</span>")


def _fold(title, body, open_=False):
    if not body:
        return ""
    return (f"<details{' open' if open_ else ''} style='margin-top:6px'><summary style='cursor:pointer;font-size:.8em;"
            f"color:#0056b3;font-weight:600'>{title}</summary><div style='padding:6px 10px;background:#fafafa;"
            f"font-size:.88em'>{body}</div></details>")


def card_html(r):
    conf = int(r["confidence_final"])
    li = lambda facts, style="": "".join(f"<li style='{style}'>{_e(f.get('fact'))} <code>{_e(f.get('source_ref'))}</code></li>" for f in facts)
    q = f"<b>{_e(_v(r, 'question'))}</b>"
    if is_false(_v(r, "is_self_contained")):
        q += f"<div style='color:#6c757d;font-size:.85em'>↳ self-contained: {_e(_v(r, 'standalone_question'))}</div>"
        q += _fold("History", _clip(_v(r, "history_text"), MAX_CHARS["history"]))
    sources = " ".join(f"<a href='{INTRAQUAL_URL.format(ref=_e(s))}' target='_blank'><code style='background:#e9ecef;"
                       f"padding:1px 5px'>{_e(s)}</code></a>" for s in r["expected_sources"])
    log_answer = _clip(_v(r, "actual_response"), MAX_CHARS["response"])
    if _v(r, "comment"):
        log_answer += "<hr><b>Comment:</b> " + _e(_v(r, "comment"))
    return f"""
<div style='border:1px solid #dee2e6;border-radius:8px;overflow:hidden;font-family:-apple-system,sans-serif;font-size:14px'>
 <div style='background:#f8f9fa;padding:8px 12px;border-bottom:1px solid #e9ecef'>
  <b style='font-family:monospace;color:#0056b3'>#{r['question_id']}</b>&nbsp;
  <span style='font-size:.75em;color:#666'>{_e(_v(r, 'source'))} · {_e(_v(r, 'slot'))} · {_e(_v(r, 'intent'))} · {_e(_v(r, 'difficulty'))}</span>&nbsp;
  {_badge(_v(r, 'final_answerability'), 'ANSWER')}{_badge(_v(r, 'ka_verdict'), 'ASSISTANT')}
  <span style='background:{ {3: "#28a745", 2: "#ffc107"}.get(conf, "#dc3545") };color:#fff;padding:2px 8px;border-radius:10px;
   font-size:.72em;font-weight:600'>confidence {conf}/3{' · EXPERT' if r['needs_expert'] else ''}</span>
 </div>
 <div style='display:grid;grid-template-columns:1fr 1fr'>
  <div style='padding:10px 12px;border-right:1px solid #f0f0f0'>
   {q}
   <div style='margin-top:8px;font-size:.75em;color:#888;font-weight:700'>ESSENTIAL FACTS (exported)</div>
   <ul style='margin:4px 0'>{li(r['essential_facts']) or '<i>none (expected behaviour: refusal / out of scope)</i>'}</ul>
   {_fold("Secondary facts", f"<ul>{li(r['secondary_facts'], 'color:#888')}</ul>" if r['secondary_facts'] else "")}
   {_fold("Facts rejected by the verifier", f"<ul>{li(r['rejected_facts'], 'color:#dc3545')}</ul>" if r['rejected_facts'] else "")}
   <div style='margin-top:6px'>{sources}</div>
   <div style='margin-top:6px;font-size:.85em;color:#555'>{"<br>".join("• " + _e(g) for g in r["guidelines"])}</div>
  </div>
  <div style='padding:10px 12px'>
   {_fold("Expected answer", _clip(_v(r, "expected_response"), MAX_CHARS["response"]), open_=True)}
   {_fold("Assistant answer (current)", _clip(_v(r, "ka_fresh_response"), MAX_CHARS["response"]))}
   {_fold("Assistant answer (logs) · vote: " + _e(_v(r, "vote") or "none"), log_answer)}
   {_fold("Relevant excerpts", _clip(_v(r, "context"), MAX_CHARS["context"]))}
   {_fold("Notes", _clip(_v(r, "notes"), MAX_CHARS["notes"]))}
  </div>
 </div>
</div>"""


class GoldReviewer:
    """A fixed number of slots created once: changing page only updates their content."""

    def __init__(self, pdf, page_size=PAGE_SIZE):
        self.pdf, self.page_size, self.page = pdf, page_size, 0
        self.dec, self._busy, self.dirty = load_decisions(), False, False
        self.dd = widgets.Dropdown(options=["all"] + STATUS + ["confidence < 2"], value="all", description="Filter:")
        self.prev = widgets.Button(description="◄", layout=widgets.Layout(width="50px"))
        self.next = widgets.Button(description="►", layout=widgets.Layout(width="50px"))
        self.save = widgets.Button(description="💾 Save", button_style="success")
        self.lbl = widgets.HTML()
        self.slots = []
        for _ in range(page_size):
            slot = {"out": widgets.Output(), "qid": None,
                    "tb": widgets.ToggleButtons(options=STATUS, value=STATUS[0], style={"button_width": "90px"}),
                    "tx": widgets.Text(placeholder="Note / correction…", layout=widgets.Layout(width="55%"))}
            slot["box"] = widgets.VBox([slot["out"], widgets.HBox([slot["tb"], slot["tx"]])], layout=widgets.Layout(margin="0 0 14px 0"))
            slot["tb"].observe(lambda _, s=slot: self._on_change(s), names="value")
            slot["tx"].observe(lambda _, s=slot: self._on_change(s), names="value")
            self.slots.append(slot)
        self.dd.observe(lambda _: self._render(reset=True), names="value")
        self.prev.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        self.save.on_click(lambda _: self._save())
        self._render(reset=True)

    def _status(self, qid):
        return self.dec.get(int(qid), ("to_review", ""))[0]

    def _rows(self):
        f = self.dd.value
        if f == "all":
            return self.pdf
        if f == "confidence < 2":
            return self.pdf[self.pdf["confidence_final"] < 2]
        return self.pdf[self.pdf["question_id"].map(self._status) == f]

    def _on_change(self, slot):
        if not self._busy and slot["qid"] is not None:
            self.dec[slot["qid"]] = (slot["tb"].value, slot["tx"].value)
            self.dirty = True
            self._label()

    def _label(self, extra=""):
        n_pages = max(1, math.ceil(len(self._rows()) / self.page_size))
        counts = Counter(self._status(q) for q in self.pdf["question_id"])
        self.lbl.value = (f"<b>Page {self.page + 1}/{n_pages}</b> — {len(self._rows())} cases · "
                          + " · ".join(f"{k}: {counts.get(k, 0)}" for k in STATUS)
                          + (" · <span style='color:#dc3545'>unsaved</span>" if self.dirty else "") + extra)

    def _move(self, delta):
        n_pages = max(1, math.ceil(len(self._rows()) / self.page_size))
        self.page = max(0, min(n_pages - 1, self.page + delta))
        self._render()

    def _render(self, reset=False):
        if reset:
            self.page = 0
        chunk = self._rows().iloc[self.page * self.page_size:(self.page + 1) * self.page_size]
        self._busy = True
        try:
            for i, slot in enumerate(self.slots):
                slot["out"].clear_output(wait=True)
                if i < len(chunk):
                    r = chunk.iloc[i]
                    slot["qid"] = int(r["question_id"])
                    slot["tb"].value, slot["tx"].value = self.dec.get(slot["qid"], ("to_review", ""))
                    with slot["out"]:
                        try:
                            ipy_display(HTML(card_html(r)))
                        except Exception as e:
                            print(f"Case #{slot['qid']} cannot be displayed: {e}")
                    slot["box"].layout.display = None
                else:
                    slot["qid"] = None
                    slot["box"].layout.display = "none"
        finally:
            self._busy = False
        self._label()

    def _save(self):
        try:
            save_decisions(self.dec)
            self.dirty = False
            self._label(" · ✓ saved")
        except Exception as e:
            self._label(f" · ❌ save failed: {html_mod.escape(str(e)[:200])}")

    def show(self):
        ipy_display(widgets.VBox([widgets.HBox([self.dd, self.prev, self.lbl, self.next, self.save],
                                               layout=widgets.Layout(align_items="center", flex_wrap="wrap")),
                                  *[s["box"] for s in self.slots]]))


pdf_final = load_golden()
print(f"{len(pdf_final)} cases to review")
GoldReviewer(pdf_final).show()

# COMMAND ----------

# DBTITLE 1,12. Export to the MLflow evaluation dataset (Unity Catalog), linked to the evaluation experiment
import os

import mlflow
import mlflow.genai.datasets as gdatasets
from databricks.sdk.runtime import display
from mlflow.entities.trace_location import UnityCatalog

REQUIRE_VALIDATION = False      # True = only cases marked "validated" in section 11
RECREATE_DATASET = True         # True = rebuild the dataset from scratch (records are otherwise only added/updated)
LANG_GUIDELINE = "Answers in the language of the user's last question."
COMPLIANCE_GUIDELINE = ("Cites the internal documents that address the requirement and says what they state; does not "
                        "assert that the company complies beyond what those documents establish.")

# Case corrections from the analysis of evaluation runs (key = unique excerpt of the conversation)
CASE_FIXES = [
    ("list me all procedures explaining work with work centers",     # list question: the expected output is the documents
     {"facts": ["Cites IN_MRPC009 (Work center management) as the procedure for managing work centers.",
                "Cites MM-1004 among the relevant procedures.", "Cites GO-1594 among the relevant procedures."],
      "sources": ["IN_MRPC009_FR", "MM-1004", "GO-1594"]}),
    ("process sheet R80", {"drop_facts_containing": ["Q0265QP", "import", "tous les sites"]}),
    ("ordre d'attribution des sites", {"drop_facts_containing": ["rule_"]}),
    ("OPEX Sharepoint", {"drop_facts_containing": ["intraqual"]}),
    ("REP_OUT and REP_FMB",                                            # not defined in the documentation: refusal case
     {"facts": [], "sources": [],
      "expected_response": "REP_OUT and REP_FMB are not defined in the available documentation; the assistant says so "
                           "and asks for clarification, without inventing a definition."}),
    ("for which plants it is applicable",
     {"reject": "reference to rebuild: the assistant cites IN-004 / QP-1232 / QP-1188, absent from the evidence"}),
    ("Access Policy", {"reject": "wrong expectation: the comparison is possible on the text given in the question"}),
    ("NF-10845 et NS-1868", {"reject": "incomplete evidence: NF-10845 was not retrieved, to regenerate"}),
    ("tolérance d'un alésage", {"reject": "to be settled by an expert: NSA2010/ABS1707 or NSA2110?"}),
]

os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

# Evaluation experiment bound to Unity Catalog trace tables (created once, reused afterwards)
if mlflow.get_experiment_by_name(EVAL_EXPERIMENT) is None:
    w.workspace.mkdirs(EVAL_EXPERIMENT.rsplit("/", 1)[0])
    mlflow.set_experiment(experiment_name=EVAL_EXPERIMENT, trace_location=UnityCatalog(
        catalog_name=TRACES_CATALOG, schema_name=TRACES_SCHEMA, table_prefix=EVAL_TRACE_PREFIX))
exp_id = mlflow.set_experiment(EVAL_EXPERIMENT).experiment_id

pdf_final = load_golden()
decisions = load_decisions()
status = lambda q: decisions.get(int(q), ("to_review", ""))[0]
to_export = pdf_final[pdf_final["question_id"].map(
    lambda q: status(q) == "validated" if REQUIRE_VALIDATION else status(q) not in ("rejected", "expert"))]


def fix_for(r):
    return next((f for key, f in CASE_FIXES if key.lower() in str(r.messages_json).lower()), {})


def clean_ref(s):
    return re.sub(r"^\s*REF\s*:\s*", "", str(s)).strip()


def to_record(r):
    fix = fix_for(r)
    facts = list(fix["facts"]) if "facts" in fix else [f["fact"] for f in r.essential_facts]
    for needle in fix.get("drop_facts_containing", []):
        facts = [f for f in facts if needle.lower() not in f.lower()]
    # The correctness judge accepts expected_facts OR expected_response, never both
    exp = {"expected_facts": facts} if facts else {"expected_response": fix.get("expected_response", r.expected_response)}
    exp["guidelines"] = [LANG_GUIDELINE, *CURATED.get(int(r.question_id), {}).get("human_guidelines", [])]
    if r.intent == "requirement_compliance":
        exp["guidelines"].append(COMPLIANCE_GUIDELINE)
    sources = [clean_ref(s) for s in fix.get("sources", r.expected_sources)]
    if sources:
        exp["expected_retrieved_context"] = [{"doc_uri": s} for s in sources]
    return {"inputs": {"messages": json.loads(r.messages_json)}, "expectations": exp}


rejected = [(int(r.question_id), fix_for(r)["reject"]) for _, r in to_export.iterrows() if "reject" in fix_for(r)]
kept = to_export[~to_export["question_id"].isin([q for q, _ in rejected])]
records = [to_record(r) for _, r in kept.iterrows()]
print(f"{len(records)} case(s) to export · {len(rejected)} set aside by CASE_FIXES:")
if len(records) < MIN_EXPORTED:
    print(f"⚠️ fewer than {MIN_EXPORTED} cases: validate more cases in section 11, or raise TARGET_N and re-run section 10.")
for q, why in rejected:
    print(f"  #{q}: {why}")
unused = [k for k, _ in CASE_FIXES if not any(k.lower() in str(m).lower() for m in to_export["messages_json"])]
if unused:
    print(f"ℹ️ corrections without a matching case in this selection: {unused}")

if RECREATE_DATASET:
    try:
        gdatasets.delete_dataset(name=EVAL_DATASET_UC)
    except Exception:
        spark.sql(f"DROP TABLE IF EXISTS {EVAL_DATASET_UC}")
try:
    eval_ds = gdatasets.get_dataset(name=EVAL_DATASET_UC)
except Exception:
    eval_ds = gdatasets.create_dataset(name=EVAL_DATASET_UC, experiment_id=exp_id)
if records:
    eval_ds.merge_records(records)
print(f"✓ {EVAL_DATASET_UC}: {len(eval_ds.to_df())} cases, linked to {EVAL_EXPERIMENT} (Datasets tab)")
display(eval_ds.to_df())

# Flat table of every reviewed case (exported or not): dataset composition and review progress for the dashboard.
# case_id matches ka_eval_results.case_id.
GOLDEN_CASES_COLUMNS = [
    ("case_id", T.StringType(), "Case id (question id of the builder; negative for synthetic questions)"),
    ("exported", T.BooleanType(), "The case is in the MLflow evaluation dataset"),
    ("exclusion_reason", T.StringType(), "Why a case is not exported: review status or correction note"),
    ("review_status", T.StringType(), "to_review, validated, rejected or expert (human review, section 11)"),
    ("review_note", T.StringType(), "Note of the human reviewer"),
    ("source", T.StringType(), "log (production conversation), override (curated log case) or synthetic"),
    ("slot", T.StringType(), "Selection slot of the shortlist (production_failure, requirement_compliance, …)"),
    ("intent", T.StringType(), "Question type"),
    ("difficulty", T.StringType(), "easy, medium or hard"),
    ("language", T.StringType(), "fr, en or other"),
    ("question", T.StringType(), "Question as asked"),
    ("final_answerability", T.StringType(), "full, partial, none or out_of_scope: whether the documentation answers it"),
    ("expected", T.StringType(), "Expected facts (one per line), or the expected answer for refusal cases"),
    ("n_expected_facts", T.LongType(), "Number of expected facts"),
    ("expected_sources", T.ArrayType(T.StringType()), "Documents the answer should rely on"),
    ("guidelines", T.ArrayType(T.StringType()), "Behavioural guidelines checked by the judges"),
    ("ka_verdict", T.StringType(), "Assistant verdict when the case was built (correct, incorrect, …)"),
    ("confidence_final", T.LongType(), "Confidence in the reference answer, 0 to 3"),
    ("dataset_name", T.StringType(), "MLflow evaluation dataset"),
    ("exported_at", T.StringType(), "Export time (UTC)"),
    ("ka_failure_stage", T.StringType(), "When the current assistant fails the case: retrieval (it retrieved none of the "
                                         "expected documents), generation (it retrieved them) or unknown; NULL otherwise"),
    ("production_error_source", T.StringType(), "Stage at fault found by the production scoring, for production failures"),
]
GOLDEN_CASES_SCHEMA = T.StructType([T.StructField(n, t) for n, t, _ in GOLDEN_CASES_COLUMNS])
exported_ids = set(kept["question_id"])
reject_notes = dict(rejected)
exported_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
case_rows = []
for _, r in pdf_final.iterrows():
    qid = int(r.question_id)
    rec = to_record(r)["expectations"] if qid in exported_ids else {}
    facts = rec.get("expected_facts", [f["fact"] for f in r.essential_facts])
    case_rows.append((
        str(qid), qid in exported_ids,
        None if qid in exported_ids else reject_notes.get(qid, f"review status: {status(qid)}"),
        status(qid), decisions.get(qid, ("", ""))[1] or None, r.source, r.slot, r.intent, r.difficulty, r.language,
        r.question, r.final_answerability, "\n".join(facts) or rec.get("expected_response", r.expected_response),
        len(facts), [clean_ref(d["doc_uri"]) for d in rec.get("expected_retrieved_context", [])] or list(r.expected_sources),
        list(rec.get("guidelines", r.guidelines)), r.ka_verdict, int(r.confidence_final), EVAL_DATASET_UC, exported_at,
        *[None if pd.isna(v) else v for v in (r.ka_failure_stage, r.production_error_source)]))
esc = lambda t: str(t).replace("'", "\\'")
if not spark.catalog.tableExists(GOLDEN_CASES_TABLE):
    spark.sql(f"CREATE TABLE {GOLDEN_CASES_TABLE} (" + ", ".join(
        f"`{n}` {t.simpleString().upper()} COMMENT '{esc(d)}'" for n, t, d in GOLDEN_CASES_COLUMNS)
        + ") COMMENT 'Qualibot golden evaluation cases, one row per reviewed case: composition and review status of the "
          "golden dataset.'")
else:
    _missing = [c for c in GOLDEN_CASES_COLUMNS if c[0] not in spark.table(GOLDEN_CASES_TABLE).columns]
    if _missing:
        spark.sql(f"ALTER TABLE {GOLDEN_CASES_TABLE} ADD COLUMNS (" + ", ".join(
            f"`{n}` {t.simpleString().upper()} COMMENT '{esc(d)}'" for n, t, d in _missing) + ")")
spark.createDataFrame(case_rows, GOLDEN_CASES_SCHEMA).createOrReplaceTempView("_golden_cases")
spark.sql(f"INSERT OVERWRITE {GOLDEN_CASES_TABLE} SELECT * FROM _golden_cases")
print(f"✓ {GOLDEN_CASES_TABLE}: {len(case_rows)} reviewed cases, {len(exported_ids)} exported")

# COMMAND ----------

# DBTITLE 1,13. Cost — estimate from the cached prompts vs actual usage (system tables)
def cost_report():
    rows = []
    for stage, outs in LLM_STAGES.items():
        t = load_stage(stage)
        if t is None:
            continue
        for o in outs:
            if f"{o}_in_chars" not in t.columns:
                continue
            a = t.agg(F.count(f"{o}_in_chars").alias("n"), F.sum(f"{o}_in_chars").alias("i"),
                      F.sum(f"{o}_out_chars").alias("o")).first()
            t_in, t_out, dbu, usd = _cost(a.i or 0, a.o or 0)
            rows.append((stage, o, a.n, round(t_in), round(t_out), round(dbu, 2), round(usd, 2)))
    rep = pd.DataFrame(rows, columns=["step", "call", "calls", "tokens_in", "tokens_out", "DBU", "USD"])
    tot = rep[["calls", "tokens_in", "tokens_out", "DBU", "USD"]].sum()
    rep.loc[len(rep)] = ["TOTAL", "", *[round(float(tot[c]), 2) for c in tot.index]]
    display(rep)
    return rep


estimate = cost_report()
me = spark.sql("SELECT current_user()").first()[0]
try:
    actual = spark.sql(f"""
        SELECT date(u.request_time) AS day, count(*) AS calls,
               sum(u.input_token_count) AS input_tokens, sum(u.output_token_count) AS output_tokens
        FROM system.serving.endpoint_usage u
        JOIN (SELECT DISTINCT served_entity_id, endpoint_name FROM system.serving.served_entities) e
          ON u.served_entity_id = e.served_entity_id
        WHERE e.endpoint_name = '{JUDGE_MODEL}' AND u.requester = '{me}' AND u.request_time >= '{USAGE_SINCE}'
        GROUP BY 1 ORDER BY 1""").toPandas()
    actual["USD"] = (actual["input_tokens"] * DBU_PER_M_INPUT + actual["output_tokens"] * DBU_PER_M_OUTPUT) / 1e6 * USD_PER_DBU
    display(actual)
    total = estimate[estimate["step"] == "TOTAL"].iloc[0]
    if actual["input_tokens"].sum() and actual["output_tokens"].sum():
        chars_per_token = total["tokens_in"] * CHARS_PER_TOKEN / actual["input_tokens"].sum()
        est_out_tokens = total["tokens_out"] * CHARS_PER_TOKEN / OUTPUT_OVERHEAD / chars_per_token
        print(f"Suggested CHARS_PER_TOKEN: {chars_per_token:.2f} · OUTPUT_OVERHEAD: "
              f"{actual['output_tokens'].sum() / est_out_tokens:.2f} (≫ 1 means hidden reasoning tokens)")
except Exception as e:
    print(f"system.serving.endpoint_usage unavailable: {str(e)[:200]}")
