"""Structured LLM calls of the golden dataset builder on a real local Spark session (needs pyspark and Java): one call
per distinct prompt, typed struct parsed from the JSON answer, rate-limited calls retried, errors kept per row."""
import pathlib
import re
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[1]
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
src = (REPO / "Build_Golden_Dataset.py").read_text()
config = src[src.index("import hashlib"):src.index("# ── Sources and storage")]
block = src[src.index("# ── Structured LLM calls"):src.index("def usage(*outs):")]

calls = []


def do(method, path, body=None):
    prompt = body["messages"][0]["content"]
    calls.append(prompt)
    assert body["response_format"]["type"] == "json_schema" and path.endswith("/databricks-gpt-6-luna/invocations")
    if "throttled" in prompt and calls.count(prompt) == 1:
        raise Exception("429 Too Many Requests: rate limit exceeded")
    if "broken" in prompt:
        raise Exception("400 Bad Request: invalid prompt")
    return {"choices": [{"message": {"content": '{"relevance": 3, "facts": ["' + prompt.upper() + '"]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


ns = {"spark": spark, "w": types.SimpleNamespace(api_client=types.SimpleNamespace(do=do)), "JUDGE_MODEL": "databricks-gpt-6-luna",
      "LLM_WORKERS": 4, "LLM_RATE_SHARE": 0.7, "LLM_INPUT_TOKENS_PER_MINUTE": 200_000, "LLM_OUTPUT_TOKENS_PER_MINUTE": 20_000,
      "LLM_MAX_ATTEMPTS": 3, "CHARS_PER_TOKEN": 3.8, "DEFAULT_OUT_CHARS": 1500}
exec(config, ns)
exec(block, ns)
ns["time"].sleep = lambda s: None                     # no real wait between retries
df = spark.createDataFrame([(1, "a", "grade a"), (1, "b", "grade a"), (2, "c", "throttled c"), (3, "d", "broken d"),
                            (4, "e", None)], "question_id long, chunk_id string, _p string")
props = {"relevance": ns["s_int"](), "facts": ns["s_arr"](ns["s_str"]())}
out = ns["llm"](df, "_p", "g", props).select("question_id", "chunk_id", "g.relevance", "g.facts", "g_error",
                                             "g_in_chars", "g_out_chars").orderBy("chunk_id").collect()
for r in out:
    print(r)
by = {r.chunk_id: r for r in out}
assert by["a"].relevance == 3 and by["a"].facts == ["GRADE A"] and by["b"].facts == ["GRADE A"]
assert calls.count("grade a") == 1, "one call per distinct prompt"
assert by["c"].facts == ["THROTTLED C"] and calls.count("throttled c") == 2, "a rate-limited call is retried"
assert by["d"].relevance is None and "400" in by["d"].g_error and calls.count("broken d") == 1, "no retry on a client error"
assert by["e"].g_error == "empty prompt"
print("builder structured calls: ok")

# What the assistant retrieved (trace returned with its answer) and the stage at fault of a failed case
import json
import math

import numpy as np
import pandas as pd

refs_block = src[src.index("_EXT = re.compile"):src.index("def code_like")]
ka_block = src[src.index("_SOURCE_HEADER = re.compile"):src.index("# COMMAND", src.index("_SOURCE_HEADER = re.compile"))]
stage_block = re.search(r"def ka_failure_stage\(.*?\n    return .*?\n", src, re.S).group(0)
ns2 = {"re": re, "json": json, "math": math, "pd": pd, "np": np, "LANG_SUFFIXES": ["FR", "GB", "BG"],
       "KA_FAIL": {"incorrect", "partially_correct", "unjustified_refusal"}}
exec(refs_block, ns2)
exec(ka_block, ns2)
exec(stage_block, ns2)
docs = [{"page_content": "[Source: QP-1457 | Title: Records] kept 10 years", "metadata": {}},
        {"page_content": "text", "metadata": {"doc_uri": "https://x/identification.aspx?ref=MI-1226_GB"}}]
raw = {"output": [], "databricks_output": {"trace": {"data": {"spans": [
    {"name": "docs", "attributes": {"mlflow.spanType": '"RETRIEVER"', "mlflow.spanOutputs": json.dumps(docs)}},
    {"name": "llm", "attributes": {"mlflow.spanType": '"LLM"', "mlflow.spanOutputs": '"answer"'}}]}}}}
assert ns2["ka_retrieval"](raw) == (["QP-1457", "MI-1226_GB"], 2), ns2["ka_retrieval"](raw)
assert ns2["ka_retrieval"]({"output": []}) == (None, None)
stage = ns2["ka_failure_stage"]
assert stage("correct", ["QP-1457"], ["QP-1457"], 2) is None
assert stage("incorrect", ["REF: QP-1457_FR"], ["QP-1457"], 2) == "generation"
assert stage("incorrect", ["Q0258MM"], ["MI-1226_GB"], 2) == "retrieval"
assert stage("unjustified_refusal", ["Q0258MM"], None, float("nan")) == "unknown"
print("assistant retrieval and stage at fault of a failed case: ok")
