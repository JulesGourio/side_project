"""Production scoring notebook end to end: 4 turns (correct answer, invented and misspelled codes, out-of-scope
refusal, answer missed although in the cited document), verdicts, output rows, MLflow traces, scorers and run.
Argument "dry" runs the dry-run path."""
import pathlib, tempfile
REPO = pathlib.Path(__file__).resolve().parents[1]
import sys, types, json, os, shutil
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import pandas as pd, mlflow
from unittest.mock import MagicMock

DRY = len(sys.argv) > 1 and sys.argv[1] == "dry"
RESET = len(sys.argv) > 1 and sys.argv[1] == "reset"   # tables already exist: reset_outputs=true replaces them
work = tempfile.mkdtemp(prefix="scoring_test_"); os.chdir(work)
mlflow.set_tracking_uri(f"sqlite:///{work}/mlflow.db")

def decide(name, text):
    if name == "question_intent": return "out_of_scope" if "courses" in text else "rule_requirement"
    if name == "answer_type":
        if "Hors p" in text: return "out_of_scope_refusal"
        if "pas trouv" in text: return "not_found"
        return "answered_partial" if "APO" in text else "answered_full"
    if name == "groundedness": return "not_supported" if "INVENTED" in text else "supported"
    if name == "missed_answer": return "yes" if "pas trouv" in text else "no"
    if name == "user_reaction": return "correction_or_complaint" if "faux" in text else "no_next_turn"
    return "yes"
calls = fake_judges(decide)

pdf = pd.DataFrame([
    dict(message_id=1, created_at=pd.Timestamp("2026-09-20"), session_id="s1", division="ALL", endpoint_name="ka-7679a56e-endpoint",
         trace_id="tr-1", answer="Selon **QP-1457**, les enregistrements sont conservés 10 ans.",
         sources_json=json.dumps([{"rank": 0, "title": "QP-1457", "url": "https://x/identification.aspx?ref=QP-1457", "n": 1}]),
         prior_messages=[{"role": "user", "content": "Durée de conservation des enregistrements ?"}], next_user_message=None,
         feedback_vote="up", feedback_comment=None),
    dict(message_id=2, created_at=pd.Timestamp("2026-09-20"), session_id="s1", division="ALL", endpoint_name="ka-7679a56e-endpoint",
         trace_id="tr-2", answer="APO: voir **IN_APO_006** et **ZZ-9999**. INVENTED claim.",
         sources_json=json.dumps([{"rank": 0, "title": "IN_APO_0006", "url": "https://x/l?id=abc", "n": 1}]),
         prior_messages=[{"role": "user", "content": "Durée ?"}, {"role": "assistant", "content": "10 ans"}, {"role": "user", "content": "que veut dire APO ?"}],
         next_user_message="c'est faux", feedback_vote="up", feedback_comment=None),
    dict(message_id=3, created_at=pd.Timestamp("2026-09-21"), session_id="s2", division="AS", endpoint_name="ka-3a7e9255-endpoint",
         trace_id=None, answer="Hors périmètre : je réponds uniquement sur la documentation qualité.", sources_json=None,
         prior_messages=[{"role": "user", "content": "fais ma liste de courses"}], next_user_message=None,
         feedback_vote=None, feedback_comment=None),
    dict(message_id=4, created_at=pd.Timestamp("2026-09-21"), session_id="s3", division="IS", endpoint_name="ka-1560aded-endpoint",
         trace_id="tr-4", answer="Je n'ai pas trouvé cette information dans **PRLAT549.FR**.", sources_json="[]",
         prior_messages=[{"role": "user", "content": "qualification du personnel LIS TUN ?"}], next_user_message=None,
         feedback_vote="down", feedback_comment="pourtant c'est dans la procédure"),
])

class FakeDF:
    def __init__(self, pdf=None, columns=()): self._pdf, self.columns = pdf, list(columns)
    def __getattr__(self, n): return lambda *a, **k: self
    @property
    def schema(self): return {"message_id": T.StructField("message_id", T.LongType()), "created_at": T.StructField("created_at", T.TimestampType())}
    def toPandas(self): return self._pdf
    def collect(self): return [types.SimpleNamespace(REF=r) for r in ["QP-1457", "IN_APO_0006", "PRLAT549.FR", "PRLAT549_GB"]]
    def count(self): return 4
sql = SparkSQL()
class Spark:
    catalog = types.SimpleNamespace(tableExists=sql.exists)
    def table(self, name):
        if name in sql.tables:
            df = FakeDF(pd.DataFrame(sql.tables[name]), columns=list(pd.DataFrame(sql.tables[name]).columns))
            df.count = lambda: len(sql.tables[name])
            return df
        return FakeDF(pdf, columns=["id", "trace_id", "content"])
    def sql(self, q, *a):
        sql.run(q)
        if "AS bad_rate" in q and q.lstrip().startswith("SELECT"):
            return FakeDF(pd.DataFrame({"endpoint_name": ["ka-7679a56e-endpoint"] * 5, "day": pd.date_range("2026-09-17", periods=5),
                                        "n": [20, 20, 20, 20, 20], "bad_rate": [0.1, 0.1, 0.1, 0.1, 0.6]}))
        return FakeDF(pdf)
    def createDataFrame(self, rows, schema): return sql.create_df(rows, schema)

w = MagicMock()
w.vector_search_indexes.get_index.return_value.delta_sync_index_spec.source_table = "uat_landingzone.qualibot.chunks_v1"
def query_index(**kw):
    refs = json.loads(kw["filters_json"])["REF"]
    return VSResult([[r, f"Excerpt of {r}: records are kept 10 years; operator qualification via QCM.", "{}"] for r in refs])
w.vector_search_indexes.query_index.side_effect = query_index
import databricks.sdk
databricks.sdk.WorkspaceClient = lambda: w

if RESET:
    for t in ["uat_proj.qualibot.chat_quality_scores", "uat_proj.qualibot.chat_quality_assessments",
              "uat_proj.qualibot.chat_quality_scoring_runs"]:
        sql.tables[t] = [{"message_id": i, "stale": True} for i in range(1, 5)]
ns = {"dbutils": types.SimpleNamespace(widgets=Widgets({"dry_run": "true" if DRY else "false", "reset_outputs": "true" if RESET else "false",
                                                         "experiment_path": "/Shared/qualibot-quality-scoring"}),
                                       library=MagicMock()),
      "spark": Spark(), "display": lambda *a, **k: print("[display]")}
run_cells(str(REPO / "Score_Production_QA.py"), ns, skip=("Setup",))

if not DRY:
    df = ns["df_final"]
    print(df[["message_id", "turn_verdict", "failure_reasons", "groundedness_level", "missed_answer", "unverified_refs", "approximate_refs", "judge_errors"]].to_string())
    print("tables:", {t: len(r) for t, r in sql.tables.items()}, "| views:", sorted(sql.views))
    if RESET:
        assert all(len(r) == n and not any(x.get("stale") for x in r) for r, n in
                   [(sql.tables["uat_proj.qualibot.chat_quality_scores"], 4), (sql.tables["uat_proj.qualibot.chat_quality_scoring_runs"], 1)])
        print("reset: stale rows replaced")
    a = pd.DataFrame(sql.tables["uat_proj.qualibot.chat_quality_assessments"])
    print(a[a["message_id"] == 2][["assessment_name", "source_type", "value", "value_numeric"]].to_string())
    ddl = next(q for q in sql.statements if q.startswith("CREATE TABLE uat_proj.qualibot.chat_quality_scores"))
    print("DDL excerpt:", ddl[:260])
    tr = mlflow.search_traces(experiment_ids=[ns["EXPERIMENT_ID"]], run_id=ns["MLFLOW_RUN_ID"], return_type="list")
    t = [x for x in tr if x.info.tags.get("message_id") == "2"][0]
    print("spans:", [s.name for s in t.data.spans], "| session:", t.info.trace_metadata.get("mlflow.trace.session"))
    print("assessments:", sorted((a.name, str(a.value)) for a in t.info.assessments))
    from mlflow.genai.scorers import list_scorers
    print("registered:", sorted(s.name for s in list_scorers(experiment_id=ns["EXPERIMENT_ID"])))
    run = mlflow.get_run(ns["MLFLOW_RUN_ID"])
    print("dataset input:", [d.dataset.name for d in run.inputs.dataset_inputs])
    print("metrics:", {k: v for k, v in run.data.metrics.items() if k.startswith(("run/bad", "reason/", "bad_rate"))})
    print("judge calls by name:", pd.Series([c[0] for c in calls]).value_counts().to_dict())
if os.environ.get("SQL_DUMP"):
    sql.dump(os.environ["SQL_DUMP"])
