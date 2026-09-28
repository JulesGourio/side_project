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

LEAKS = []   # judges other than user_reaction must never see the user's next message


def decide(name, text):
    if name != "user_reaction" and "c'est faux" in text:
        LEAKS.append(name)
    if name == "question_intent": return "out_of_scope" if "courses" in text else "rule_requirement"
    if name == "answer_type":
        if "Hors p" in text: return "out_of_scope_refusal"
        if "pas trouv" in text: return "not_found"
        return "answered_partial" if "APO" in text else "answered_full"
    if name == "groundedness": return "not_supported" if "INVENTED" in text else "supported"
    if name == "missed_answer": return "yes" if "pas trouv" in text else "no"
    if name == "user_reaction": return "correction_or_complaint" if "faux" in text else "no_next_turn"
    if name == "retrieval_quality": return "none" if "(no passage)" in text else "full"
    if name == "compliance_claim": return "unsupported_compliance_claim" if "INVENTED" in text else "not_applicable"
    return "yes"
calls = fake_judges(decide)

pdf = pd.DataFrame([
    dict(message_id=1, created_at=pd.Timestamp("2026-09-20"), session_id="s1", division="ALL", endpoint_name="ka-7679a56e-endpoint",
         trace_id="tr-1", answer="Selon **QP-1457**, les enregistrements sont conservés 10 ans.⟦1⟧\n\nLes enregistrements "
         "papier sont archivés par le service qualité.⟦1⟧ [QP-1457](https://x/identification.aspx?ref=QP-1457"
         "#:~:text=records%20are%20kept%2010%20years)",
         sources_json=json.dumps([{"rank": 0, "title": "QP-1457", "url": "https://x/identification.aspx?ref=QP-1457", "n": 1}]),
         prior_messages=[{"role": "user", "content": "Durée de conservation des enregistrements ?"}], next_user_message=None,
         feedback_vote="up", feedback_comment=None),
    dict(message_id=2, created_at=pd.Timestamp("2026-09-20"), session_id="s1", division="ALL", endpoint_name="ka-7679a56e-endpoint",
         trace_id="tr-2", answer="APO: voir **IN_APO_006** et **ZZ-9999**. INVENTED claim.",
         sources_json=json.dumps([{"rank": 0, "title": "IN_APO_0006", "url": "https://x/l?id=abc", "n": 1}]),
         prior_messages=[{"role": "user", "content": "Durée ?"}, {"role": "assistant", "content": "10 ans"}, {"role": "user", "content": "que veut dire APO ?"}],
         next_user_message="c'est faux. " + "La réponse ne cite pas la bonne exigence du client Dassault. " * 8,
         feedback_vote="up", feedback_comment=None),
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
        if "information_schema.views" in q:
            return types.SimpleNamespace(collect=lambda: [{"table_name": "v_chat_quality_daily"},
                                                          {"table_name": "other_view"}])
        if "AS bad_rate" in q and q.lstrip().startswith("SELECT"):
            return FakeDF(pd.DataFrame({"endpoint_name": ["ka-7679a56e-endpoint"] * 5, "day": pd.date_range("2026-09-17", periods=5),
                                        "n": [20, 20, 20, 20, 20], "bad_rate": [0.1, 0.1, 0.1, 0.1, 0.6]}))
        return FakeDF(pdf)
    def createDataFrame(self, rows, schema): return sql.create_df(rows, schema)

# The assistants' own traces: turn 1 shows its retrieval step, turn 4 a retrieval step that returned nothing (the
# independent search of the index then finds the answer), turn 2 has no trace (excerpts of the cited documents instead)
class KATrace:
    def __init__(self, tid, docs):
        self.info = types.SimpleNamespace(trace_id=tid)
        self._d = {"info": {}, "data": {"spans": [
            {"name": "agent", "attributes": {"mlflow.spanType": '"AGENT"', "mlflow.spanOutputs": '"answer"'}},
            {"name": "vector_search", "attributes": {"mlflow.spanType": '"RETRIEVER"',
                                                     "mlflow.spanInputs": '{"query": "q"}',
                                                     "mlflow.spanOutputs": json.dumps(docs)}}]}}
    def to_dict(self): return self._d
KA_TRACES = {"tr-1": KATrace("tr-1", [{"page_content": "[Source: QP-1457 | Title: Records] Records are kept 10 years.",
                                       "metadata": {"doc_uri": "https://x/identification.aspx?ref=QP-1457"}}]),
             "tr-4": KATrace("tr-4", [])}
_get_trace = mlflow.get_trace
mlflow.get_trace = lambda tid, *a, **k: KA_TRACES[tid] if tid in KA_TRACES else _get_trace(tid, *a, **k)

w = MagicMock()
w.vector_search_indexes.get_index.return_value.delta_sync_index_spec.source_table = "uat_landingzone.qualibot.chunks_v1"
def query_index(**kw):
    if not kw.get("filters_json"):                       # independent search of the whole index
        return VSResult([["PRLAT549_GB", "Corpus chunk: LIS TUN personnel qualification by QCM.", "{}"]])
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

queries = ns["TURNS"]["1"]["evidence_queries"]
print("evidence queries of turn 1:", queries)
assert list(queries) == ["QP-1457"] and len(queries["QP-1457"]) == 4, queries        # question, 2 citing lines, quote
assert "records are kept 10 years" in queries["QP-1457"], queries
assert all("⟦" not in q and "http" not in q for q in queries["QP-1457"]), queries
assert set(ns["TURNS"]["2"]["evidence_queries"]) == {"IN_APO_0006"}, ns["TURNS"]["2"]["evidence_queries"]

if not DRY:
    df = ns["df_final"]
    print(df[["message_id", "turn_verdict", "failure_reasons", "groundedness_level", "missed_answer", "unverified_refs", "approximate_refs", "judge_errors"]].to_string())
    print("tables:", {t: len(r) for t, r in sql.tables.items()})
    assert not sql.views, "the dashboard reads the tables: no view is created"
    assert "DROP VIEW IF EXISTS uat_proj.qualibot.v_chat_quality_daily" in sql.statements
    assert not any("other_view" in q for q in sql.statements if q.startswith("DROP"))
    ev = df.set_index(df["message_id"].astype(str))
    print(ev[["evidence_source", "evidence_count", "retrieval_quality", "compliance_claim", "error_source",
              "judge_input_tokens", "judge_cost_usd"]].to_string())
    assert ev.loc["1", "evidence_source"] == "assistant_retrieval" and ev.loc["1", "evidence_count"] == 1
    assert ev.loc["1", "retrieval_quality"] == "sufficient"
    assert ev.loc["2", "evidence_source"] == "cited_document_excerpts" and pd.isna(ev.loc["2", "retrieval_quality"])
    assert ev.loc["4", "retrieval_quality"] == "retrieval_miss" and "retrieval_miss" in ev.loc["4", "failure_reasons"]
    assert ev.loc["2", "error_source"] == "generation" and ev.loc["4", "error_source"] == "retrieval"
    assert ev["judge_input_tokens"].min() > 0
    compliance_turn = {"answer_type": "answered_full", "question_intent": "requirement_compliance", "relevance": "yes",
                       "groundedness": "not_supported", "retrieval_quality": "retrieval_miss", "citation_count": 2}
    assert ns["turn_verdict"](compliance_turn)[::2] == ("bad", "retrieval_and_generation")
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
    long_tags = {k: len(v) for k, v in (t.info.tags or {}).items() if not k.startswith("mlflow.") and len(v) > 250}
    assert not long_tags, f"trace tags longer than 250 characters: {long_tags}"
    print("trace tags all within 250 characters: ok")
    print("assessments:", sorted((a.name, str(a.value)) for a in t.info.assessments))
    from mlflow.genai.scorers import list_scorers
    print("registered:", sorted(s.name for s in list_scorers(experiment_id=ns["EXPERIMENT_ID"])))
    run = mlflow.get_run(ns["MLFLOW_RUN_ID"])
    print("dataset input:", [d.dataset.name for d in run.inputs.dataset_inputs])
    print("metrics:", {k: v for k, v in run.data.metrics.items() if k.startswith(("run/bad", "reason/", "bad_rate"))})
    a_rows = pd.DataFrame(sql.tables["uat_proj.qualibot.chat_quality_assessments"])
    expected = a_rows[a_rows["assessment_name"] == "groundedness"]["value_numeric"].astype(float).mean()
    assert abs(run.data.metrics["groundedness/mean"] - expected) < 1e-3, (run.data.metrics.get("groundedness/mean"), expected)
    assert "turn_verdict/mean" in run.data.metrics
    print("scorer means over the whole run: ok")
    assert not LEAKS, f"next user message visible to: {sorted(set(LEAKS))}"
    print("next user message hidden from the other judges: ok")
    print("judge calls by name:", pd.Series([c[0] for c in calls]).value_counts().to_dict())
if os.environ.get("SQL_DUMP"):
    sql.dump(os.environ["SQL_DUMP"])
