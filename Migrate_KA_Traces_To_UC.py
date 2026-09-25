# Databricks notebook source
# MAGIC %md
# MAGIC # Qualibot — Knowledge Assistant Trace Migration to Unity Catalog
# MAGIC
# MAGIC Copies the MLflow traces of the Knowledge Assistants (experiments `ka-…-dev-experiment`) into the
# MAGIC Unity Catalog trace tables `<catalog>.<schema>.<prefix>_otel_{spans,annotations,logs,metrics}`.
# MAGIC Designed to run as a scheduled job (Databricks Asset Bundle, service principal); it can also be run interactively.
# MAGIC
# MAGIC - Requires classic compute on **DBR 15.3 or above** (serverless is not supported by the migration).
# MAGIC - Source experiments are **never modified**.
# MAGIC - The migration is **idempotent**: each run only copies traces not yet migrated, so a nightly schedule acts as an
# MAGIC   incremental synchronisation.
# MAGIC
# MAGIC Job parameters (`base_parameters`): `to_migrate` (comma-separated table prefixes, or `*` for the three assistants),
# MAGIC `migrate_last_days` (empty = full history), `repair_destinations`, `traces_catalog`, `traces_schema`, `experiment_dir`.

# COMMAND ----------

# MAGIC %pip install -qqq "databricks-agents>=1.10.1" "mlflow[databricks]>=3.1"

# COMMAND ----------

# CELL M.0
dbutils.library.restartPython()

# COMMAND ----------

# CELL M.1 — Configuration and parameters
import time
import mlflow
import pyspark.sql.functions as F
from mlflow import MlflowClient
from mlflow.entities.trace_location import UnityCatalog
from databricks.sdk import WorkspaceClient

dbutils.widgets.text("to_migrate", "trace_test", "Prefixes to migrate (comma-separated) or *")
dbutils.widgets.text("migrate_last_days", "", "Last N days only (empty = full history)")
dbutils.widgets.dropdown("repair_destinations", "true", ["true", "false"], "Create / repair destinations")
dbutils.widgets.text("traces_catalog", "uat_proj", "Catalog of the trace tables")
dbutils.widgets.text("traces_schema", "qualibot", "Schema of the trace tables")
dbutils.widgets.text("experiment_dir", "/Workspace/Users/jules.gourio.external@latecoere.aero/qualibot-traces",
                     "Folder of the destination experiments")

TRACES_CATALOG = dbutils.widgets.get("traces_catalog").strip()
TRACES_SCHEMA = dbutils.widgets.get("traces_schema").strip()
TRACES_EXPERIMENT_DIR = dbutils.widgets.get("experiment_dir").strip().rstrip("/")
OTEL_SUFFIXES = ["otel_spans", "otel_annotations", "otel_logs", "otel_metrics"]

# Destination table prefix -> source experiment of the assistant
KA_TRACE_SOURCES = {
    "trace_ka_all_v2": "4171178917767011",   # qualibot_ALL_v2 (ka-7679a56e)
    "trace_ka_is_v2": "2748374992560665",    # qualibot_IS_v2  (ka-1560aded)
    "trace_ka_as_v2": "2748374992560664",    # qualibot_AS_v2  (ka-3a7e9255)
}
TEST_SOURCES = {"trace_test": "3375946803618197"}   # ka-99026e27: disposable assistant used to validate the job
ALL_SOURCES = {**KA_TRACE_SOURCES, **TEST_SOURCES}

_raw = dbutils.widgets.get("to_migrate").strip()
TO_MIGRATE = list(KA_TRACE_SOURCES) if _raw == "*" else [p.strip() for p in _raw.split(",") if p.strip()]
unknown = [p for p in TO_MIGRATE if p not in ALL_SOURCES]
if unknown:
    raise ValueError(f"Unknown prefixes: {unknown}. Allowed values: {list(ALL_SOURCES)} or *")
_days = dbutils.widgets.get("migrate_last_days").strip()
MIGRATE_LAST_DAYS = int(_days) if _days else None
REPAIR = dbutils.widgets.get("repair_destinations") == "true"

client, w = MlflowClient(), WorkspaceClient()
ME = spark.sql("SELECT current_user()").first()[0]
print(f"Identity : {ME}")
print(f"To migrate: {TO_MIGRATE} · window: {MIGRATE_LAST_DAYS or 'full history'} · repair: {REPAIR}")

try:
    print(f"Runtime  : {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")
except Exception:
    raise RuntimeError("Runtime not detected (serverless?): use classic compute on DBR 15.3 or above.")

# COMMAND ----------

# CELL M.1b — Create missing destinations and repair those without trace tables
def tables_present(prefix):
    return sum(spark.catalog.tableExists(f"{TRACES_CATALOG}.{TRACES_SCHEMA}.{prefix}_{s}") for s in OTEL_SUFFIXES)

if REPAIR:
    w.workspace.mkdirs(TRACES_EXPERIMENT_DIR)
    for prefix in TO_MIGRATE:
        name = f"{TRACES_EXPERIMENT_DIR}/{prefix}"
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None and tables_present(prefix) == 4:
            print(f"✓ {prefix}: ready")
            continue
        if exp is not None:   # experiment created but not bound to Unity Catalog: set it aside
            client.rename_experiment(exp.experiment_id, f"{name}_broken_{int(time.time())}")
            client.delete_experiment(exp.experiment_id)
            print(f"🗑️ {prefix}: unbound experiment {exp.experiment_id} renamed and moved to trash")
        new = mlflow.set_experiment(
            experiment_name=name,
            trace_location=UnityCatalog(catalog_name=TRACES_CATALOG, schema_name=TRACES_SCHEMA, table_prefix=prefix),
        )
        print(f"✓ {prefix}: created (id {new.experiment_id}), {tables_present(prefix)}/4 tables")

# COMMAND ----------

# CELL M.2 — Pre-flight checks (the job fails here if a destination is not ready)
TARGETS, problems = {}, []
for prefix in TO_MIGRATE:
    src = mlflow.get_experiment(ALL_SOURCES[prefix])
    dst = mlflow.get_experiment_by_name(f"{TRACES_EXPERIMENT_DIR}/{prefix}")
    n_tables = tables_present(prefix)
    print(f"{prefix}\n  source      : {src.name} (id {src.experiment_id})"
          f"\n  destination : {(dst.name + ' (id ' + dst.experiment_id + ')') if dst else 'not found'}"
          f"\n  UC tables   : {n_tables}/4")
    if dst is None or n_tables < 4:
        problems.append(prefix)
    else:
        TARGETS[prefix] = dst.experiment_id
if problems:
    raise RuntimeError(f"Destinations not ready: {problems} (run again with repair_destinations=true)")

# COMMAND ----------

# CELL M.3 — Explicit SELECT + MODIFY grants for the running identity (required by the migration)
for prefix in TO_MIGRATE:
    for s in OTEL_SUFFIXES:
        table = f"{TRACES_CATALOG}.{TRACES_SCHEMA}.{prefix}_{s}"
        try:
            spark.sql(f"GRANT SELECT, MODIFY ON TABLE {table} TO `{ME}`")
        except Exception as e:
            print(f"⚠️ GRANT not applied on {table}: {str(e)[:120]}")
print("Grants checked.")

# COMMAND ----------

# CELL M.4 — Migration
from databricks.migrations.migrate_traces_to_uc import run

kwargs = {}
if MIGRATE_LAST_DAYS:
    kwargs["start_time_ms"] = int((time.time() - MIGRATE_LAST_DAYS * 24 * 3600) * 1000)

failures = {}
for prefix in TO_MIGRATE:
    print(f"→ {prefix}: {ALL_SOURCES[prefix]} → {TARGETS[prefix]}")
    t0 = time.time()
    try:
        run(source_experiment_id=ALL_SOURCES[prefix], target_experiment_id=TARGETS[prefix], **kwargs)
        print(f"✓ {prefix} migrated in {time.time() - t0:.0f} s")
    except Exception as e:
        failures[prefix] = str(e)
        print(f"❌ {prefix}: {str(e)[:300]}")

# COMMAND ----------

# CELL M.5 — Verification (the job fails if any migration failed)
for prefix in TO_MIGRATE:
    spans = f"{TRACES_CATALOG}.{TRACES_SCHEMA}.{prefix}_otel_spans"
    try:
        n = spark.table(spans).select(F.countDistinct("trace_id")).first()[0]
        print(f"{prefix}: {n} traces in {spans}")
    except Exception as e:
        print(f"{prefix}: count failed ({str(e)[:150]})")

skipped = spark.sql(f"SHOW TABLES IN {TRACES_CATALOG}.{TRACES_SCHEMA} LIKE '*migration_skipped*'").collect()
for t in skipped:
    print(f"⚠️ skipped traces are listed in {TRACES_CATALOG}.{TRACES_SCHEMA}.{t.tableName}")
    display(spark.table(f"{TRACES_CATALOG}.{TRACES_SCHEMA}.{t.tableName}").limit(20))

if failures:
    raise RuntimeError(f"Migration failed for: {list(failures)}")
print("✓ Migration completed.")
