"""Runs Databricks notebook cells locally with stubs for dbutils, Spark, Vector Search and the judge model."""
import os, sys, types, json, re
from unittest.mock import MagicMock
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# ── Minimal pyspark stub ──
class _T:
    def __init__(self, *a, **k): self.a = a
    def __repr__(self): return type(self).__name__
class StructField:
    def __init__(self, name, dataType, nullable=True): self.name, self.dataType = name, dataType
class StructType:
    def __init__(self, fields=None): self.fields = fields or []
    def __getitem__(self, k): return next(f for f in self.fields if f.name == k)
class ArrayType(_T):
    def __init__(self, el): self.elementType = el
T = types.ModuleType("pyspark.sql.types")
for n in ["BooleanType", "DoubleType", "LongType", "StringType", "TimestampType", "IntegerType"]:
    setattr(T, n, type(n, (_T,), {}))
T.StructField, T.StructType, T.ArrayType = StructField, StructType, ArrayType
pyspark = types.ModuleType("pyspark"); sql = types.ModuleType("pyspark.sql")
sql.types = T; sql.functions = MagicMock(); pyspark.sql = sql
sys.modules.update({"pyspark": pyspark, "pyspark.sql": sql, "pyspark.sql.types": T,
                    "pyspark.sql.functions": sql.functions, "pyspark.sql.window": MagicMock()})

class Widgets:
    def __init__(self, values): self.v = dict(values)
    def text(self, name, default, *a): self.v.setdefault(name, default)
    def dropdown(self, name, default, choices, *a): self.v.setdefault(name, default)
    def get(self, name): return self.v[name]

def cells(path):
    src = open(path).read()
    out = []
    for c in src.split("# COMMAND ----------"):
        title = re.search(r"# DBTITLE 1,(.*)", c)
        out.append((title.group(1) if title else c.strip().split("\n")[0], c))
    return out

def run_cells(path, ns, only=None, skip=()):
    for title, code in cells(path):
        if "# MAGIC" in code and "%md" in code:
            continue
        if any(s in title for s in skip):
            continue
        if only and not any(o in title for o in only):
            continue
        print(f"── cell: {title[:90]}")
        import linecache
        fname = f"<cell {os.path.basename(path)}::{title[:40]}>"
        linecache.cache[fname] = (len(code), None, code.splitlines(True), fname)
        exec(compile(code, fname, "exec"), ns)

def fake_judges(decide):
    """decide(assessment_name, prompt_text) -> value."""
    from mlflow.entities import Feedback, AssessmentSource
    import mlflow.genai.judges.instructions_judge as ij, mlflow.genai.judges.builtin as bj
    calls = []
    def fake(model_uri, prompt, assessment_name, **kw):
        text = prompt if isinstance(prompt, str) else "\n".join(getattr(m, "content", str(m)) for m in prompt)
        calls.append((assessment_name, model_uri))
        return Feedback(name=assessment_name, value=decide(assessment_name, text), rationale=f"fake {assessment_name}",
                        source=AssessmentSource(source_type="LLM_JUDGE", source_id=str(model_uri)))
    ij.invoke_judge_model = fake; bj.invoke_judge_model = fake
    return calls

class VSResult:
    def __init__(self, rows, cols=("REF", "chunk_text", "semantic_headers")):
        self.manifest = types.SimpleNamespace(columns=[types.SimpleNamespace(name=c) for c in cols])
        self.result = types.SimpleNamespace(data_array=rows)
