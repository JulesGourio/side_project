"""Replays on a real local Spark session the SQL written by the notebooks in tests/test_scoring.py and
tests/test_eval.py (run with SQL_DUMP=<file>): table DDL with comments and inserted rows, then typical dashboard
queries on the tables. Delta-only statements (MERGE) are skipped. Needs pyspark and Java.

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
for d in dumps:
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
assert not any(q.startswith("CREATE OR REPLACE VIEW") for d in dumps for q in d["statements"]), "no view expected"
DASHBOARD_QUERIES = {
    "daily quality": """SELECT DATE(created_at) AS day, endpoint_name, COUNT(*) AS n,
                               AVG(IF(turn_verdict = 'bad', 1.0, 0.0)) AS bad_rate, SUM(estimated_cost_usd) AS cost
                        FROM qualibot.chat_quality_scores WHERE turn_verdict IS NOT NULL GROUP BY ALL""",
    "failure reasons": """SELECT reason, COUNT(*) AS n
                          FROM qualibot.chat_quality_scores LATERAL VIEW explode(failure_reasons) r AS reason GROUP BY reason""",
    "scorers over time": """SELECT DATE(created_at) AS day, assessment_name, AVG(value_numeric) AS mean_value
                            FROM qualibot.chat_quality_assessments GROUP BY ALL""",
    "evaluation metrics": """SELECT started_at, endpoint, subset, scorers_config_id, metric, score, ci_low, ci_high
                             FROM qualibot.ka_eval_metrics""",
    "shared scorers": """SELECT 'production' AS context, assessment_name, AVG(value_numeric) AS mean_value
                         FROM qualibot.chat_quality_assessments
                         WHERE assessment_name IN ('relevance', 'groundedness', 'missed_answer') GROUP BY ALL
                         UNION ALL
                         SELECT 'evaluation', assessment_name, AVG(value_numeric)
                         FROM qualibot.ka_eval_assessments
                         WHERE assessment_name IN ('relevance', 'groundedness', 'missed_answer') GROUP BY ALL""",
}
for name, q in DASHBOARD_QUERIES.items():
    if all(spark.catalog.tableExists(t) for t in re.findall(r"qualibot\.\w+", q)):
        out = spark.sql(q)
        print(f"✓ {name}: {out.count()} rows · columns {out.columns}")
t = spark.sql("DESCRIBE TABLE EXTENDED qualibot.chat_quality_scores").filter("col_name = 'Comment'").collect() \
    if spark.catalog.tableExists("qualibot.chat_quality_scores") else []
print("table comment:", t[0].data_type[:80] if t else "n/a")
print("column comment:", spark.sql("DESCRIBE TABLE qualibot.ka_eval_results").filter("col_name = 'groundedness'")
      .collect()[0].comment if spark.catalog.tableExists("qualibot.ka_eval_results") else "n/a")
