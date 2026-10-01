# Databricks notebook source
# DBTITLE 1,Export Lakebase (latlang) to UC volume
import datetime
import json
import os
from decimal import Decimal

import psycopg2
import psycopg2.extras
from databricks.sdk import WorkspaceClient

# Serverless job task (no cluster spec) — classic new_cluster compute can't
# reach the Lakebase private endpoint in this workspace.
dbutils.widgets.text("LAKEBASE_PROJECT_ID", "qualibot")
dbutils.widgets.text("LAKEBASE_BRANCH", "production")
dbutils.widgets.text("LAKEBASE_ENDPOINT", "primary")
dbutils.widgets.text("LAKEBASE_DATABASE", "latlang")
dbutils.widgets.text("OUTPUT_VOLUME_CATALOG", "uat_landingzone")
dbutils.widgets.text("OUTPUT_VOLUME_SCHEMA", "latlang")
dbutils.widgets.text("OUTPUT_VOLUME_NAME", "latlang")
dbutils.widgets.text("TABLES_TO_SKIP", "")

LAKEBASE_PROJECT_ID = dbutils.widgets.get("LAKEBASE_PROJECT_ID")
LAKEBASE_BRANCH = dbutils.widgets.get("LAKEBASE_BRANCH")
LAKEBASE_ENDPOINT = dbutils.widgets.get("LAKEBASE_ENDPOINT")
LAKEBASE_DATABASE = dbutils.widgets.get("LAKEBASE_DATABASE")
OUTPUT_VOLUME_CATALOG = dbutils.widgets.get("OUTPUT_VOLUME_CATALOG")
OUTPUT_VOLUME_SCHEMA = dbutils.widgets.get("OUTPUT_VOLUME_SCHEMA")
OUTPUT_VOLUME_NAME = dbutils.widgets.get("OUTPUT_VOLUME_NAME")
TABLES_TO_SKIP = {
    table.strip()
    for table in dbutils.widgets.get("TABLES_TO_SKIP").split(",")
    if table.strip()
}

OUTPUT_DIR = f"/Volumes/{OUTPUT_VOLUME_CATALOG}/{OUTPUT_VOLUME_SCHEMA}/{OUTPUT_VOLUME_NAME}/lakebase_export"


def json_serializer(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (bytes, memoryview)):
        return obj.hex() if isinstance(obj, bytes) else bytes(obj).hex()
    return str(obj)


spark.sql(
    f"CREATE VOLUME IF NOT EXISTS `{OUTPUT_VOLUME_CATALOG}`.`{OUTPUT_VOLUME_SCHEMA}`.`{OUTPUT_VOLUME_NAME}`"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

w = WorkspaceClient()
branch_path = f"projects/{LAKEBASE_PROJECT_ID}/branches/{LAKEBASE_BRANCH}"
endpoint_path = f"{branch_path}/endpoints/{LAKEBASE_ENDPOINT}"

endpoints = list(w.postgres.list_endpoints(parent=branch_path))
if not endpoints:
    raise RuntimeError(f"Aucun endpoint trouvé pour {branch_path}")

endpoint = next(
    (ep for ep in endpoints if getattr(ep, "name", None) == LAKEBASE_ENDPOINT),
    endpoints[0],
)
lakebase_host = endpoint.status.hosts.host

me = w.current_user.me()
username = me.user_name or me.display_name
if not username:
    raise RuntimeError("Impossible de résoudre l'identité courante.")

credential = w.postgres.generate_database_credential(endpoint=endpoint_path)
if not credential.token:
    raise RuntimeError("generate_database_credential a retourné un token vide.")

conn = psycopg2.connect(
    host=lakebase_host,
    port=5432,
    database=LAKEBASE_DATABASE,
    user=username,
    password=credential.token,
    sslmode="require",
)

try:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
            """
        )
        tables = cur.fetchall()

    print(
        f"Export Lakebase {LAKEBASE_PROJECT_ID}/{LAKEBASE_BRANCH}/{LAKEBASE_DATABASE} -> {OUTPUT_DIR}"
    )
    print(f"{len(tables)} table(s) trouvée(s)")

    for schema, table in tables:
        if table in TABLES_TO_SKIP:
            print(f"  {schema}.{table}... ignorée (TABLES_TO_SKIP)")
            continue

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f'SELECT * FROM "{schema}"."{table}"')
            rows = [dict(row) for row in cur.fetchall()]

        out_file = f"{OUTPUT_DIR}/{table}.json"
        with open(out_file, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, default=json_serializer) + "\n"
                )

        print(f"  {schema}.{table}: {len(rows)} ligne(s) -> {out_file}")
finally:
    conn.close()

print(f"Terminé. Export disponible dans {OUTPUT_DIR}")


# COMMAND ----------
