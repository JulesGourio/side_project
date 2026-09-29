"""Step cache of the golden dataset builder on a real local Spark session (needs pyspark and Java): a cached row is
recomputed only when it is missing, failed, or its inputs or configuration changed; rows cached without a fingerprint
are recomputed once. Also checks the process sheet route and the meta-fact rule."""
import hashlib
import json
import pathlib
import re
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import types as T

spark = (SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
         .config("spark.sql.legacy.createHiveTableByDefault", "false")
         .config("spark.sql.warehouse.dir", tempfile.mkdtemp()).getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
src = (REPO / "Build_Golden_Dataset.py").read_text()
block = src[src.index("# ── Step cache ──"):src.index("# ── Structured LLM calls")]

CACHE = "qualibot_eval_cache"
spark.sql(f"CREATE TABLE {CACHE} (stage STRING, question_id BIGINT, chunk_id STRING, payload STRING, "
          "payload_schema STRING, updated_at TIMESTAMP) USING PARQUET")


def delete_where(keep):
    """Local stand-in for the Delta DELETE / MERGE of the notebook: rewrites the cache without the deleted rows."""
    rows = [r for r in spark.table(CACHE).collect() if keep(r)]
    spark.createDataFrame(rows, spark.table(CACHE).schema).write.mode("overwrite").saveAsTable(CACHE + "_tmp")
    spark.sql(f"INSERT OVERWRITE {CACHE} SELECT * FROM {CACHE}_tmp")


ns = {"spark": spark, "F": F, "T": T, "json": json, "hashlib": hashlib, "CACHE_TABLE": CACHE, "FORCE": set(),
      "CHUNK_STAGES": set(), "ESTIMATE_ONLY": set(), "LLM_KEYS_PER_WRITE": 2, "_DRY": [False]}
exec(block, ns)
ns["_delete_stage"] = lambda stage: delete_where(lambda r: r.stage != stage)
ns["_delete_keys"] = lambda stage, kdf, chunk_level=False: delete_where(
    lambda r, keys={row.question_id for row in kdf.collect()}: not (r.stage == stage and r.question_id in keys))

calls = []


def build(todo):
    ids = [r.question_id for r in todo.collect()]
    calls.extend(ids)
    return todo.select("question_id", F.upper("q").alias("answer"))


def run(rows, config="prompt v", depends_on=("q",)):
    calls.clear()
    df = spark.createDataFrame(rows, "question_id long, q string, other string")
    out = ns["incremental"]("annotations", df, ["question_id"], build, ok_col="answer",
                            depends_on=list(depends_on) if depends_on else None, config=config)
    assert "input_hash" not in out.columns, "the fingerprint stays inside the cache"
    return sorted(calls), {r.question_id: r.answer for r in out.collect()}


rows = [(1, "a", "x"), (2, "b", "x"), (3, "c", "x")]
assert run(rows, depends_on=None) == ([1, 2, 3], {1: "A", 2: "B", 3: "C"})
assert run(rows)[0] == [1, 2, 3], "rows cached without a fingerprint are recomputed once"
assert run(rows)[0] == [], "nothing changed: nothing recomputed"
assert run([(1, "a", "y"), (2, "b", "y"), (3, "c", "y")])[0] == [], "a column the step does not read is ignored"
assert run([(1, "a", "x"), (2, "B2", "x"), (3, "c", "x")]) == ([2], {1: "A", 2: "B2", 3: "C"}), "changed input"
assert run([(1, "a", "x"), (2, "B2", "x"), (3, "c", "x"), (4, "d", "x")])[0] == [4], "new key"
assert run([(1, "a", "x"), (2, "B2", "x"), (3, "c", "x"), (4, "d", "x")], config="prompt w")[0] == [1, 2, 3, 4], \
    "a new prompt recomputes every row"
print("step cache: ok")

# Driver-side steps (Vector Search, assistant): same rules
py_calls = []


def compute(rows_):
    py_calls.extend(r["question_id"] for r in rows_)
    return [(r["question_id"], "c1", r["q"] * 2) for r in rows_]


def run_py(rows_):
    py_calls.clear()
    df = spark.createDataFrame(rows_, "question_id long, q string")
    out = ns["incremental_py"]("evidence_pool", df, compute, "question_id long, chunk_id string, text string",
                               depends_on=["q"], config="routes")
    return sorted(py_calls), {r.question_id: r.text for r in out.collect()}


assert run_py([(1, "a"), (2, "b")]) == ([1, 2], {1: "aa", 2: "bb"})
assert run_py([(1, "a"), (2, "b")])[0] == []
assert run_py([(1, "a"), (2, "z")]) == ([2], {1: "aa", 2: "zz"})
print("driver-side step cache: ok")

# Process sheets named by their process code, and statements about the excerpts
refs_block = src[src.index("_EXT = re.compile"):src.index("def vs_source_table")]
ns2 = {"re": re, "LANG_SUFFIXES": ["FR", "GB", "EN"]}
exec(refs_block, ns2)
exec(src[src.index('_PROCESS_CODE = re.compile'):src.index("def refs_in_sources_json")], ns2)
catalog = {}
for ref in ["PRO-S40", "PRO-S40-E", "PROLAT_P28_FR", "PROLAT_P28_EN", "PROLAT_P30_FR", "QP-1457", "PRO-R80"]:
    catalog.setdefault(ns2["base_ref"](ref), set()).add(ref)
ns2["load_ref_catalog"] = lambda: catalog
got = ns2["process_sheet_refs"]("PEux me donner le rapport entre le process S40 et le processus P28")
assert got == ["PRO-S40", "PRO-S40-E", "PROLAT_P28_EN", "PROLAT_P28_FR"], got
assert ns2["process_sheet_refs"]("tolérance A350, norme ISO 18490") == []
assert ns2["plain_ref"]("REF: IN-PLANNING-002") == "IN-PLANNING-002"
meta = ns2["META_FACT"]
assert meta.search("The sequence is not present in the excerpts.") and meta.search("Les extraits ne définissent pas X")
assert not meta.search("APO means Analyste Performance Opérationnelle.")
print("process sheets, plain codes, meta facts: ok")
