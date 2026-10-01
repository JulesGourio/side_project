# Databricks notebook source
# MAGIC %md
# MAGIC # Grant Volume Access
# MAGIC
# MAGIC One-off admin utility: grants USE CATALOG + USE SCHEMA + READ/WRITE VOLUME
# MAGIC on a single volume to a service principal. Copied from latec-compare's
# MAGIC identical script (same pattern: run_as a service principal that already
# MAGIC has grant authority on the target catalog).
# MAGIC
# MAGIC Use case here: declaring a bundle `apps.*.resources` uc_securable block
# MAGIC makes Terraform try to auto-grant USE CATALOG to `account users`, which
# MAGIC fails unless the deploying user has MANAGE on the catalog. This job
# MAGIC grants the app's own service principal directly instead, after the app
# MAGIC has been created without that resources block.
# MAGIC
# MAGIC If `run_as` also lacks GRANT rights on the target catalog, this job fails
# MAGIC with the same PERMISSION_DENIED the target service principal hits — a job
# MAGIC can't manufacture a privilege `run_as` doesn't already have.

# COMMAND ----------

dbutils.widgets.text("service_principal", "", "Service principal client id (or account) to grant")
dbutils.widgets.text("catalog", "", "Catalog (e.g. prod_landingzone)")
dbutils.widgets.text("schema_name", "", "Schema (e.g. intraqual)")
dbutils.widgets.text("volume_name", "", "Volume (e.g. intraqual_documents)")
dbutils.widgets.text("volume_permission", "READ VOLUME", "Unused — kept for job-parameter compatibility; both READ and WRITE are always granted (an app needs both to download originals and upload outputs)")

service_principal  = dbutils.widgets.get("service_principal").strip()
catalog            = dbutils.widgets.get("catalog").strip()
schema_name        = dbutils.widgets.get("schema_name").strip()
volume_name        = dbutils.widgets.get("volume_name").strip()

assert service_principal, "service_principal widget is required"
assert catalog, "catalog widget is required"
assert schema_name, "schema_name widget is required"
assert volume_name, "volume_name widget is required"

print(f"Granting USE CATALOG + USE SCHEMA + READ/WRITE VOLUME on "
      f"{catalog}.{schema_name}.{volume_name} to `{service_principal}`")

# COMMAND ----------

statements = [
    f"GRANT USE CATALOG ON CATALOG `{catalog}` TO `{service_principal}`",
    f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema_name}` TO `{service_principal}`",
    f"GRANT READ VOLUME ON VOLUME `{catalog}`.`{schema_name}`.`{volume_name}` TO `{service_principal}`",
    f"GRANT WRITE VOLUME ON VOLUME `{catalog}`.`{schema_name}`.`{volume_name}` TO `{service_principal}`",
]

failed = []
for stmt in statements:
    try:
        spark.sql(stmt)
        print(f"  OK: {stmt}")
    except Exception as exc:
        failed.append((stmt, str(exc).splitlines()[0][:300]))
        print(f"  FAILED: {stmt}\n    -> {str(exc).splitlines()[0][:300]}")

if failed:
    raise RuntimeError(
        f"{len(failed)}/{len(statements)} grant statement(s) failed — run_as lacks "
        f"GRANT rights on {catalog}.{schema_name}; a real Unity Catalog "
        f"metastore/catalog admin needs to run this instead."
    )

print("All grants applied successfully.")
