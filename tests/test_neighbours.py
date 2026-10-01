"""Neighbour expansion of the golden dataset builder on a real local Spark session (needs pyspark and Java)."""
import pathlib, tempfile
REPO = pathlib.Path(__file__).resolve().parents[1]
import re, types, hashlib, sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
src = (REPO / "Build_Golden_Dataset.py").read_text()
fn = re.search(r"def compute_neighbours\(rows\):.*?return list\(out.values\(\)\)\n", src, re.S).group(0)
def chunk_key(text): return hashlib.md5(re.sub(r"\s+", " ", str(text).strip()).encode()).hexdigest()
spark.createDataFrame([
    (1, "PRLAT506", 0, "Scope of the procedure", "h0"), (1, "PRLAT506", 1, "Operators answer a QCM", "h1"),
    (1, "PRLAT506", 2, "Exception: temporary staff", "h2"), (1, "PRLAT506", 3, "Annex", "h3"),
    (2, "QP-1457", 0, "Records kept 10 years", "h"), (2, "QP-1457", 1, "Unless the customer requires more", "h")],
    "IDDOC long, REF string, chunk_index int, chunk_text string, semantic_headers string").createOrReplaceTempView("chunks_v1")
ns = {"spark": spark, "F": F, "chunk_key": chunk_key, "NEIGHBOUR_WINDOW": 1, "vs_source_table": lambda: "chunks_v1"}
exec(fn, ns)
rows = [types.SimpleNamespace(question_id=10, best_chunks=[{"REF": "PRLAT506", "chunk_text": "Operators answer a QCM"}],
                              known_ids=[chunk_key("Scope of the procedure")]),
        types.SimpleNamespace(question_id=11, best_chunks=[{"REF": "QP-1457", "chunk_text": "Records kept 10 years"}], known_ids=[])]
for r in sorted(ns["compute_neighbours"](rows)): print(r[0], r[2], r[4], r[5])
