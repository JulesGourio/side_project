"""Golden dataset builder: the flat table ka_eval_golden_cases written at export, on a real local Spark session
(needs pyspark, pandas and Java); two exports in a row leave one row per case."""
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
import re, tempfile, types, json
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.types as T
spark = (SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
         .config("spark.sql.legacy.createHiveTableByDefault", "false")
         .config("spark.sql.warehouse.dir", tempfile.mkdtemp()).getOrCreate())
spark.sparkContext.setLogLevel("ERROR"); spark.sql("CREATE DATABASE IF NOT EXISTS qualibot")
src = (REPO / "Build_Golden_Dataset.py").read_text()
block = src[src.index("# Flat table of every reviewed case"):src.index('print(f"✓ {GOLDEN_CASES_TABLE}')] + 'print(f"✓ {GOLDEN_CASES_TABLE}: {len(case_rows)} reviewed cases, {len(exported_ids)} exported")'
pdf_final = pd.DataFrame([
    dict(question_id=-1, source="synthetic", slot="curated", intent="definition_acronym", difficulty="easy", language="fr",
         question="que signifie l'acronyme APO ?", final_answerability="full", expected_response="APO means ...",
         essential_facts=[{"fact": "APO means Analyste Performance Opérationnelle."}], expected_sources=["IN_APO_0006"],
         guidelines=["Must cite IN_APO_0006"], ka_verdict="incorrect", confidence_final=3, ka_failure_stage="retrieval",
         production_error_source=None),
    dict(question_id=42, source="log", slot="requirement_compliance", intent="requirement_compliance", difficulty="hard", language="fr",
         question="Latécoère est-il conforme à l'exigence Dassault 4.2 ?", final_answerability="partial", expected_response="...",
         essential_facts=[], expected_sources=["QP-1457"], guidelines=[], ka_verdict="incorrect", confidence_final=1,
         ka_failure_stage=float("nan"), production_error_source="generation")])
ns = dict(spark=spark, T=T, pd=pd, pdf_final=pdf_final, GOLDEN_CASES_TABLE="qualibot.ka_eval_golden_cases",
          EVAL_DATASET_UC="uat_landingzone.qualibot.qualibot_eval_golden",
          kept=pdf_final[pdf_final.question_id == -1], rejected=[], decisions={42: ("expert", "ask a quality expert")},
          status=lambda q: {-1: "validated", 42: "expert"}[q],
          to_record=lambda r: {"expectations": {"expected_facts": [f["fact"] for f in r.essential_facts],
                                                "expected_retrieved_context": [{"doc_uri": "IN_APO_0006"}], "guidelines": ["g"]}},
          clean_ref=lambda s: re.sub(r"^\s*REF\s*:\s*", "", str(s)).strip())
exec(block, ns); exec(block, ns)       # twice: the second run overwrites
spark.table("qualibot.ka_eval_golden_cases").select("case_id", "exported", "exclusion_reason", "review_status", "intent", "n_expected_facts", "expected_sources").show(truncate=40)
print(spark.sql("DESCRIBE TABLE qualibot.ka_eval_golden_cases").filter("col_name='exported'").collect()[0].comment)
stages = {r.case_id: (r.ka_failure_stage, r.production_error_source) for r in
          spark.table("qualibot.ka_eval_golden_cases").select("case_id", "ka_failure_stage", "production_error_source").collect()}
assert stages == {"-1": ("retrieval", None), "42": (None, "generation")}, stages
print("stage at fault columns:", stages)
