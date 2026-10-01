# Databricks notebook source
# MAGIC %md
# MAGIC # App Lifecycle (start / stop)
# MAGIC
# MAGIC Starts or stops a Databricks App via the SDK. Backs the
# MAGIC `start_app`/`stop_app` scheduled jobs (weekday-only cron) so the app
# MAGIC isn't billed as `RUNNING` outside business hours / on weekends.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("app_name", "", "App name")
dbutils.widgets.dropdown("action", "start", ["start", "stop"], "Action")

app_name = dbutils.widgets.get("app_name").strip()
action = dbutils.widgets.get("action").strip()

assert app_name, "app_name widget is required"

w = WorkspaceClient()

print(f"{action}ing app '{app_name}'...")
if action == "start":
    w.apps.start(name=app_name).result()
else:
    w.apps.stop(name=app_name).result()

print(f"App '{app_name}' {action} completed.")
