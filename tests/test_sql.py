"""Replays on a real local Spark session the SQL written by the notebooks in tests/test_scoring.py and
tests/test_eval.py (run with SQL_DUMP=<file>): table DDL with comments, inserted rows, dashboard views, then queries
every view. Delta-only statements (MERGE) are skipped. Needs pyspark and Java.

    SQL_DUMP=/tmp/scoring.json python tests/test_scoring.py && SQL_DUMP=/tmp/eval.json python tests/test_eval.py ""
    python tests/test_sql.py /tmp/scoring.json /tmp/eval.json
"""
import json
import re
import sys
import tempfile

from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
         .config("spark.sql.legacy.createHiveTableByDefault", "false")
         .config("spark.sql.warehouse.dir", tempfile.mkdtemp(prefix="spark_warehouse_")).getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.sql("CREATE DATABASE IF NOT EXISTS qualibot")
local = lambda q: re.sub(r"\buat_proj\.qualibot\.", "qualibot.", q)

dumps = [json.load(open(p)) for p in sys.argv[1:]]
for d in dumps:                                   # tables first: the views of one notebook read the other's tables
    for q in d["statements"]:
        if q.startswith("CREATE TABLE"):
            spark.sql(local(q))
    for table, rows in d["tables"].items():
        if not rows:
            continue
        name = local(table + ".")[:-1]
        schema = spark.table(name).schema
        df = spark.read.json(spark.sparkContext.parallelize([json.dumps(r, default=str) for r in rows]))
        cols = [f"CAST({f.name} AS {f.dataType.simpleString()}) AS {f.name}" if f.name in df.columns
                else f"CAST(NULL AS {f.dataType.simpleString()}) AS {f.name}" for f in schema.fields]
        df.selectExpr(*cols).write.insertInto(name)
for d in dumps:
    for q in d["statements"]:
        if q.startswith("CREATE OR REPLACE VIEW"):
            spark.sql(local(q))
            view = re.match(r"CREATE OR REPLACE VIEW (\S+)", local(q)).group(1)
            out = spark.table(view)
            print(f"✓ {view}: {out.count()} rows · columns {out.columns}")
t = spark.sql("DESCRIBE TABLE EXTENDED qualibot.chat_quality_scores").filter("col_name = 'Comment'").collect() \
    if spark.catalog.tableExists("qualibot.chat_quality_scores") else []
print("table comment:", t[0].data_type[:80] if t else "n/a")
print("column comment:", spark.sql("DESCRIBE TABLE qualibot.ka_eval_results").filter("col_name = 'groundedness'")
      .collect()[0].comment if spark.catalog.tableExists("qualibot.ka_eval_results") else "n/a")
