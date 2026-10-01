"""The shared cell is identical in both notebooks; citation text fragments are removed from answers; numeric forms of
the verdicts."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def shared_cell(notebook: str) -> str:
    cells = (REPO / notebook).read_text().split("# COMMAND ----------")
    return next(c for c in cells if "# DBTITLE 1,Shared scorers and helpers" in c).strip()


evaluation, production = shared_cell("Evaluate_Knowledge_Assistant.py"), shared_cell("Score_Production_QA.py")
assert evaluation == production, "the shared cell differs between the two notebooks"
print("shared cell identical in both notebooks:", len(production.splitlines()), "lines")

ns = {}
exec("import re\n_TEXT_FRAGMENT" + production.split("_TEXT_FRAGMENT", 1)[1].split("\n\n\nCONTEXT")[0], ns)
answer = ("Ce suivi se fait à l'aide de l'outil «Qualification opérateur».[^eOF0-1]\n\n[^eOF0-1]: Vérification de la "
          "connaissance des règles. [https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=Q0196QP_FR]"
          "(https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=Q0196QP_FR#:~:text=V%C3%A9rification"
          "%20de%20la%20connaissance%20des%20r%C3%A8gles%0ACe%20suivi%20se%20fait%20%C3%A0%20l%E2%80%99aide)")
cleaned = ns["clean_answer"](answer)
assert "#:~:text=" not in cleaned and cleaned.endswith("?ref=Q0196QP_FR)"), cleaned
print(f"clean_answer: {len(answer)} → {len(cleaned)} characters, link kept")

block = production[production.index("SCORE_VALUES = "):production.index("def assessment_row")]
ns = {}
exec(block, ns)
nv = ns["numeric_value"]
checks = {("relevance", "yes"): 1.0, ("missed_answer", "yes"): 0.0, ("missed_answer", "no"): 1.0,
          ("groundedness", "partially_supported"): 0.5, ("fact_coverage", "none"): 0.0, ("fact_coverage", None): None,
          ("reference_integrity", False): 0.0, ("citation_count", 3): 3.0, ("question_intent", "document_lookup"): None,
          ("turn_verdict", "acceptable"): 0.5, ("user_vote", "down"): 0.0}
bad = {k: (nv(*k), v) for k, v in checks.items() if nv(*k) != v}
assert not bad, bad
print("numeric_value: all", len(checks), "checks pass")

# The assistant's retrieval is read from a real MLflow trace (Trace object) and from its JSON form
import json
import mlflow
from mlflow.entities import Document

mlflow.set_tracking_uri("sqlite:///" + __import__("tempfile").mkdtemp() + "/mlflow.db")
block = production[production.index("_SOURCE_HEADER = re.compile"):production.index("def record_assistant_retrieval")]
ns = {"re": re, "json": json, "Document": Document}
exec(block, ns)


@mlflow.trace(span_type="RETRIEVER")
def vector_search(query):
    return [Document(page_content="[Source: QP-1457 | Title: Records] Records are kept 10 years.",
                     metadata={"doc_uri": "https://intraqual/identification.aspx?ref=QP-1457"}),
            Document(page_content="Chunk without header.", metadata={"doc_uri": "https://intraqual/x.aspx?ref=Q0196QP_FR"})]


@mlflow.trace(span_type="AGENT")
def agent(query):
    vector_search(query)
    return "answer"


agent("retention?")
getattr(mlflow, "flush_trace_async_logging", lambda: None)()          # traces are written asynchronously
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
for form in (trace, json.loads(json.dumps(trace.to_dict()))):
    steps, passages = ns["retrieved_passages"](form)
    assert steps == 1 and [p.metadata["doc_uri"] for p in passages] == ["QP-1457", "Q0196QP_FR"], (steps, passages)
print("retrieved_passages: Trace object and JSON form, document codes resolved")
