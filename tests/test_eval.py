"""Evaluation notebook end to end on a 3-case golden dataset: assistant with url_citation annotations, judges,
report, dataset links. Argument: sample size ("" = full dataset)."""
import pathlib, tempfile
REPO = pathlib.Path(__file__).resolve().parents[1]
import sys, types, json, os, shutil
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import pandas as pd, mlflow
from unittest.mock import MagicMock

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "3"
work = tempfile.mkdtemp(prefix="eval_test_"); os.chdir(work)
mlflow.set_tracking_uri(f"sqlite:///{work}/mlflow.db")
EXP = "/Workspace/Users/x/qualibot-traces/trace_eval_all_v2"
exp_id = mlflow.create_experiment(EXP)
import mlflow.genai.datasets as gd
ds = gd.create_dataset(name="uat_landingzone.qualibot.qualibot_eval_golden", experiment_id=exp_id)
ds.merge_records([
    {"inputs": {"messages": [{"role": "user", "content": "que signifie l'acronyme APO ?"}]},
     "expectations": {"expected_facts": ["APO means Analyste Performance Opérationnelle."], "guidelines": ["Answers in the language of the user's last question."],
                      "expected_retrieved_context": [{"doc_uri": "IN_APO_0006"}]}},
    {"inputs": {"messages": [{"role": "user", "content": "fais moi ma liste de courses pour ce week end"}]},
     "expectations": {"expected_response": "Out of scope.", "guidelines": ["Must politely decline the request."]}},
    {"inputs": {"messages": [{"role": "user", "content": "qualification du personnel LIS TUN ?"}]},
     "expectations": {"expected_facts": ["Competences are managed in the GDC tool."], "expected_retrieved_context": [{"doc_uri": "REF: PRLAT549_FR"}]}},
])

def decide(name, text):
    return {"fact_coverage": "full", "fact_contradiction": "no_contradiction", "refusal_handling": "not_applicable",
            "groundedness": "partially_supported", "missed_answer": "no"}.get(name, "yes")
calls = fake_judges(decide)

def ka(method, path, body=None):
    q = (body.get("input") or body.get("messages"))[-1]["content"]
    if "courses" in q:
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hors périmètre."}]}]}
    if "APO" in q:
        text = "APO signifie Analyste Performance Opérationnelle (**IN_APO_006**), voir aussi **ZZ-9999**."
        url = "https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=IN_APO_0006#:~:text=APO"
    else:
        text = "Voir **PRLAT549.FR** : outil GDC."
        url = "https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=PRLAT549.FR#:~:text=GDC"
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text,
            "annotations": [{"type": "url_citation", "title": url, "url": url}]}]}]}
w = MagicMock()
w.api_client.do.side_effect = ka
w.vector_search_indexes.get_index.return_value.delta_sync_index_spec.source_table = "uat_landingzone.qualibot.chunks_v1"
w.vector_search_indexes.query_index.side_effect = lambda **kw: VSResult(
    [[r, f"Excerpt of {r}: APO = Analyste Performance Opérationnelle; GDC tool.", "{}"] for r in json.loads(kw["filters_json"])["REF"]])
import databricks.sdk
databricks.sdk.WorkspaceClient = lambda: w
rt = types.ModuleType("databricks.sdk.runtime")
rt.display = lambda *a, **k: print("[display]", (a[0].to_string()[:1500] if hasattr(a[0], "to_string") else ""))
sys.modules["databricks.sdk.runtime"] = rt

sql = SparkSQL()
sql.tables["uat_proj.qualibot.chat_quality_assessments"] = []   # written by the production scoring
class Spark:
    catalog = types.SimpleNamespace(tableExists=sql.exists)
    def table(self, name):
        if "eval_cache" in name: raise Exception("no cache table")
        if name in sql.tables:
            df = MagicMock(); df.columns = list(pd.DataFrame(sql.tables[name]).columns); return df
        df = MagicMock(); df.select.return_value.distinct.return_value.collect.return_value = [
            types.SimpleNamespace(REF=r) for r in ["IN_APO_0006", "PRLAT549.FR", "PRLAT549_GB", "QP-1457"]]
        return df
    def sql(self, q, *a):
        sql.run(q); return MagicMock()
    def createDataFrame(self, rows, schema): return sql.create_df(rows, schema)
ns = {"dbutils": types.SimpleNamespace(widgets=Widgets({"run_eval": "true", "sample_n": SAMPLE, "experiment_path": EXP}), library=MagicMock()),
      "spark": Spark(), "display": rt.display}
run_cells(str(REPO / "Evaluate_Knowledge_Assistant.py"), ns,
          skip=("Setup", "Run comparison", "Human labels", "Judge alignment", "Review export"))
res = ns["collect"](ns["RUN_IDS"][-1])
print(res.drop(columns=["_why", "_tags", "_assessments", "expectations", "answer", "trace_id", "outputs", "tags", "source"], errors="ignore").T.to_string()[:3000])
run = mlflow.get_run(ns["RUN_IDS"][-1])
print("dataset inputs:", [d.dataset.name for d in run.inputs.dataset_inputs])
print("judge models used:", sorted({c[1] for c in calls}))
print("tables:", {t: len(r) for t, r in sql.tables.items()}, "| views:", sorted(sql.views))
print(pd.DataFrame(sql.tables["uat_proj.qualibot.ka_eval_metrics"])[["metric", "score", "ci_low", "ci_high", "n"]].to_string())
r = pd.DataFrame(sql.tables["uat_proj.qualibot.ka_eval_results"])
print(r[["case_id", "question", "correctness", "groundedness", "reference_integrity", "failed_scorers", "expected_sources"]].to_string())
print("clean answer:", r.loc[r.question.str.contains("APO"), "answer"].iloc[0])
if os.environ.get("SQL_DUMP"):
    sql.dump(os.environ["SQL_DUMP"])
