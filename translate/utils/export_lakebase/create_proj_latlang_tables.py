# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Lakebase → Unity Catalog table copy
# MAGIC
# MAGIC Copies every table of a Lakebase (Postgres) database into Delta tables of a Unity Catalog schema, in one pass and
# MAGIC without any intermediate file or volume: rows are streamed from Postgres with a server-side cursor and written
# MAGIC directly with Spark.
# MAGIC
# MAGIC ### Behaviour
# MAGIC - **Generic**: tables and columns are discovered from the Postgres catalog at every run; nothing is hardcoded.
# MAGIC   Works with Lakebase Autoscaling (project / branch / endpoint) and Lakebase Provisioned (database instance).
# MAGIC - **Typed**: Delta column types are derived from the Postgres types (integers, decimals, booleans, dates,
# MAGIC   timestamps with and without time zone, arrays, binary); JSON, UUID, enums and any other type become `STRING`.
# MAGIC   No type inference, so an all-NULL column never changes type from one run to the next.
# MAGIC - **Tables created when needed**: the target schema (optional) and every target table are created if missing.
# MAGIC   Each table is fully replaced at every run (`CREATE OR REPLACE`, atomic, Delta history kept), including empty
# MAGIC   tables, so a schema change in Postgres is reflected without manual action.
# MAGIC - **Consistent**: all tables are read in a single read-only `REPEATABLE READ` transaction (same snapshot).
# MAGIC - **Robust**: a failing table does not stop the others; the run fails at the end if any table failed.
# MAGIC - Tables removed from Postgres are not removed from Unity Catalog.
# MAGIC
# MAGIC ### Requirements
# MAGIC - **Serverless compute** (classic clusters may not reach the Lakebase private endpoint).
# MAGIC - The identity running the notebook (job `run_as`) needs a Postgres role on the Lakebase database with `SELECT` on
# MAGIC   the source tables, and `USE CATALOG`, `USE SCHEMA` / `CREATE SCHEMA`, `CREATE TABLE` on the target, plus
# MAGIC   ownership of target tables that already exist.
# MAGIC
# MAGIC ### Parameters
# MAGIC | Widget | Meaning |
# MAGIC |---|---|
# MAGIC | `lakebase_project` | Lakebase Autoscaling project id (leave empty when `lakebase_instance` is used) |
# MAGIC | `lakebase_branch` | Branch of the project (default `production`) |
# MAGIC | `lakebase_endpoint` | Endpoint of the branch; empty = the branch's first endpoint |
# MAGIC | `lakebase_instance` | Lakebase Provisioned database instance name (alternative to the project) |
# MAGIC | `lakebase_host` | Optional host override (otherwise resolved from the project or the instance) |
# MAGIC | `lakebase_database` | Postgres database (default `databricks_postgres`) |
# MAGIC | `source_schemas` | Comma-separated Postgres schemas; empty = every non-system schema |
# MAGIC | `include_tables` | Comma-separated `table` or `schema.table`; empty = every table |
# MAGIC | `skip_tables` | Comma-separated `table` or `schema.table` to skip |
# MAGIC | `target_catalog`, `target_schema` | Unity Catalog destination |
# MAGIC | `create_target_schema` | `true` = `CREATE SCHEMA IF NOT EXISTS` before copying |
# MAGIC | `target_table_name` | Target name pattern, placeholders `{schema}` and `{table}` (e.g. `lb_{table}`, `{schema}_{table}`) |
# MAGIC | `fetch_batch_rows` | Rows fetched per round trip to Postgres |

# COMMAND ----------

# DBTITLE 1,Setup — installs only missing packages, without altering the runtime's own packages
import importlib.metadata as md, subprocess, sys

# Package (distribution names accepted) -> minimum version. databricks-sdk >= 0.102 provides the Lakebase
# Autoscaling API (`w.postgres`).
NEEDED = {"psycopg2-binary": ((2, 9), ("psycopg2-binary", "psycopg2")), "databricks-sdk": ((0, 102), ("databricks-sdk",))}


def _as_tuple(text):
    return tuple(int(x) for x in text.split(".")[:2] if x.isdigit())


def _installed(names):
    for name in names:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    return None


missing = [f"{pkg}>={'.'.join(map(str, version))}" for pkg, (version, names) in NEEDED.items()
           if not _installed(names) or _as_tuple(_installed(names)) < version]
if missing:
    # Every package already present is pinned (except the ones being installed): pip either adds what is missing or
    # fails explicitly, and cannot downgrade core packages such as protobuf.
    upgraded = {name for _, names in NEEDED.values() for name in names}
    pins = [l for l in subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().splitlines()
            if "==" in l and l.split("==")[0].lower() not in upgraded]
    with open("/tmp/pinned_packages.txt", "w") as f:
        f.write("\n".join(pins))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/pinned_packages.txt", *missing])
    dbutils.library.restartPython()
print({pkg: _installed(names) for pkg, (_, names) in NEEDED.items()})

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("lakebase_project", "", "Lakebase project id (Autoscaling)")
dbutils.widgets.text("lakebase_branch", "production", "Lakebase branch")
dbutils.widgets.text("lakebase_endpoint", "", "Lakebase endpoint (empty = first)")
dbutils.widgets.text("lakebase_instance", "", "Lakebase instance (Provisioned)")
dbutils.widgets.text("lakebase_host", "", "Host override (optional)")
dbutils.widgets.text("lakebase_database", "databricks_postgres", "Postgres database")
dbutils.widgets.text("source_schemas", "", "Postgres schemas (empty = all)")
dbutils.widgets.text("include_tables", "", "Tables to copy (empty = all)")
dbutils.widgets.text("skip_tables", "", "Tables to skip")
dbutils.widgets.text("target_catalog", "", "Target catalog")
dbutils.widgets.text("target_schema", "", "Target schema")
dbutils.widgets.dropdown("create_target_schema", "true", ["true", "false"], "Create target schema")
dbutils.widgets.text("target_table_name", "{table}", "Target table name pattern")
dbutils.widgets.text("fetch_batch_rows", "20000", "Rows per fetch")

PARAMS = {name: dbutils.widgets.get(name).strip() for name in [
    "lakebase_project", "lakebase_branch", "lakebase_endpoint", "lakebase_instance", "lakebase_host",
    "lakebase_database", "source_schemas", "include_tables", "skip_tables", "target_catalog", "target_schema",
    "create_target_schema", "target_table_name", "fetch_batch_rows"]}

# COMMAND ----------

# DBTITLE 1,Pure functions (no Spark / Databricks dependency)
import datetime
import decimal
import json
import re

SYSTEM_SCHEMAS = {"information_schema"}          # plus every schema starting with "pg_"

# Postgres type (as returned by format_type, without length / precision) -> Spark SQL type
SIMPLE_TYPES = {
    "smallint": "SMALLINT", "integer": "INT", "bigint": "BIGINT",
    "real": "FLOAT", "double precision": "DOUBLE",
    "boolean": "BOOLEAN", "date": "DATE",
    "timestamp with time zone": "TIMESTAMP", "timestamp without time zone": "TIMESTAMP_NTZ",
    "bytea": "BINARY",
}
DECIMAL_MAX_PRECISION = 38


def parse_list(text):
    """Comma-separated parameter -> list of non-empty stripped items."""
    return [item.strip() for item in (text or "").split(",") if item.strip()]


def is_system_schema(schema):
    return schema in SYSTEM_SCHEMAS or schema.startswith("pg_")


def matches(schema, table, selectors):
    """True if `table` or `schema.table` is in the selectors (case-insensitive)."""
    keys = {s.lower() for s in selectors}
    return table.lower() in keys or f"{schema}.{table}".lower() in keys


def map_pg_type(pg_type):
    """Postgres type as returned by format_type() (e.g. 'numeric(10,2)', 'integer[]') -> Spark SQL type."""
    pg_type = pg_type.strip().lower()
    if pg_type.endswith("[]"):
        return f"ARRAY<{map_pg_type(pg_type[:-2])}>"
    base = re.sub(r"\(.*?\)", "", pg_type).strip()    # 'timestamp(3) with time zone' -> 'timestamp with time zone'
    if base in SIMPLE_TYPES:
        return SIMPLE_TYPES[base]
    if base == "numeric":
        args = re.search(r"\((\d+)\s*(?:,\s*(-?\d+))?\)", pg_type)
        if not args:
            return "DOUBLE"                          # unconstrained numeric: no fixed precision available
        precision, scale = int(args.group(1)), int(args.group(2) or 0)
        if precision > DECIMAL_MAX_PRECISION or scale < 0 or scale > precision:
            return "STRING"                          # not representable as a Spark decimal: kept exact as text
        return f"DECIMAL({precision},{scale})"
    return "STRING"                                  # text, varchar, uuid, json(b), enums, time, interval, inet…


def _to_text(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)


def make_converter(spark_type):
    """Spark SQL type -> function converting a psycopg2 value into a value accepted by that Spark type."""
    if spark_type.startswith("ARRAY<"):
        element = make_converter(spark_type[len("ARRAY<"):-1])
        return lambda v: None if v is None else [element(x) for x in v]
    if spark_type == "STRING":
        return lambda v: None if v is None else _to_text(v)
    if spark_type == "BINARY":
        return lambda v: None if v is None else bytes(v)
    if spark_type in ("FLOAT", "DOUBLE"):
        return lambda v: None if v is None else float(v)
    if spark_type in ("SMALLINT", "INT", "BIGINT"):
        return lambda v: None if v is None else int(v)
    if spark_type.startswith("DECIMAL"):
        return lambda v: None if v is None else decimal.Decimal(v)
    return lambda v: v                               # BOOLEAN, DATE, TIMESTAMP, TIMESTAMP_NTZ: native Python types


def quote_ident(name):
    """Spark identifier quoting."""
    return "`" + name.replace("`", "``") + "`"


def spark_ddl(columns):
    """[(column name, Spark type)] -> DDL schema string for spark.createDataFrame."""
    return ", ".join(f"{quote_ident(name)} {spark_type}" for name, spark_type in columns)


def resolve_target_names(tables, pattern):
    """[(schema, table)] -> {(schema, table): target table name}; raises on collisions or an invalid pattern."""
    if "{table}" not in pattern:
        raise ValueError(f"target_table_name must contain {{table}}: {pattern!r}")
    names = {(s, t): pattern.format(schema=s, table=t) for s, t in tables}
    seen = {}
    for key, name in names.items():
        if name.lower() in seen:
            raise ValueError(f"Target name collision: {seen[name.lower()]} and {key} both map to {name!r}. "
                             "Use a pattern containing {schema}, e.g. '{schema}_{table}', or skip one of them.")
        seen[name.lower()] = key
    return names

# COMMAND ----------

# DBTITLE 1,Configuration checks
TARGET_CATALOG, TARGET_SCHEMA = PARAMS["target_catalog"], PARAMS["target_schema"]
if not TARGET_CATALOG or not TARGET_SCHEMA:
    raise ValueError("target_catalog and target_schema are required.")
if not (PARAMS["lakebase_project"] or PARAMS["lakebase_instance"] or PARAMS["lakebase_host"]):
    raise ValueError("Set lakebase_project (Autoscaling), lakebase_instance (Provisioned) or lakebase_host.")
if PARAMS["lakebase_project"] and PARAMS["lakebase_instance"]:
    raise ValueError("Set either lakebase_project or lakebase_instance, not both.")

SOURCE_SCHEMAS = parse_list(PARAMS["source_schemas"])
INCLUDE_TABLES = parse_list(PARAMS["include_tables"])
SKIP_TABLES = parse_list(PARAMS["skip_tables"])
FETCH_BATCH_ROWS = int(PARAMS["fetch_batch_rows"] or 20000)
TARGET_PREFIX = f"{quote_ident(TARGET_CATALOG)}.{quote_ident(TARGET_SCHEMA)}"

# COMMAND ----------

# DBTITLE 1,Connection to Lakebase (identity of the notebook / job, short-lived OAuth token)
import uuid

import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me()
PG_USER = me.user_name or me.display_name      # service principals: application id
if not PG_USER:
    raise RuntimeError("Could not resolve the current identity.")

if PARAMS["lakebase_instance"]:
    # Lakebase Provisioned
    instance = w.database.get_database_instance(name=PARAMS["lakebase_instance"])
    host = PARAMS["lakebase_host"] or instance.read_write_dns
    token = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[PARAMS["lakebase_instance"]]).token
    source_label = f"instance {PARAMS['lakebase_instance']}"
else:
    # Lakebase Autoscaling
    branch_path = f"projects/{PARAMS['lakebase_project']}/branches/{PARAMS['lakebase_branch'] or 'production'}"
    endpoints = list(w.postgres.list_endpoints(parent=branch_path))
    if not endpoints:
        raise RuntimeError(f"No endpoint found for {branch_path}")
    wanted = PARAMS["lakebase_endpoint"]
    endpoint = next((ep for ep in endpoints
                     if wanted and (ep.name or "").split("/")[-1] == wanted.split("/")[-1]), None)
    if wanted and endpoint is None:
        raise RuntimeError(f"Endpoint {wanted!r} not found in {[ep.name for ep in endpoints]}")
    endpoint = endpoint or endpoints[0]
    host = PARAMS["lakebase_host"] or endpoint.status.hosts.host
    token = w.postgres.generate_database_credential(endpoint=endpoint.name).token
    source_label = endpoint.name

if not token:
    raise RuntimeError("Lakebase returned an empty database credential.")

conn = psycopg2.connect(host=host, port=5432, dbname=PARAMS["lakebase_database"], user=PG_USER, password=token,
                        sslmode="require", application_name="lakebase_to_uc_copy")
# One read-only snapshot for every table: related tables (e.g. messages and feedbacks) stay consistent.
conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
print(f"Connected to {source_label} / database {PARAMS['lakebase_database']} as {PG_USER}")

# COMMAND ----------

# DBTITLE 1,Discovery of the source tables and their columns
with conn.cursor() as cur:
    # Ordinary and partitioned tables; partitions themselves are excluded (their rows are read through the parent).
    cur.execute("""
        SELECT c.oid, n.nspname, c.relname
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p') AND NOT c.relispartition
        ORDER BY n.nspname, c.relname
    """)
    all_tables = cur.fetchall()

selected = [(oid, s, t) for oid, s, t in all_tables
            if not is_system_schema(s)
            and (not SOURCE_SCHEMAS or s in SOURCE_SCHEMAS)
            and (not INCLUDE_TABLES or matches(s, t, INCLUDE_TABLES))
            and not matches(s, t, SKIP_TABLES)]
skipped = sorted({f"{s}.{t}" for _, s, t in all_tables if not is_system_schema(s)}
                 - {f"{s}.{t}" for _, s, t in selected})
if not selected:
    raise RuntimeError(f"No table selected. Tables available: {[f'{s}.{t}' for _, s, t in all_tables]}")

TARGET_NAMES = resolve_target_names([(s, t) for _, s, t in selected], PARAMS["target_table_name"] or "{table}")

COLUMNS = {}
with conn.cursor() as cur:
    for oid, schema, table in selected:
        # Domains are resolved to their base type so that e.g. a domain over integer stays an integer.
        cur.execute("""
            SELECT a.attname,
                   format_type(CASE WHEN t.typtype = 'd' THEN t.typbasetype ELSE a.atttypid END,
                               CASE WHEN t.typtype = 'd' THEN t.typtypmod ELSE a.atttypmod END)
            FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = %s AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum
        """, (oid,))
        COLUMNS[(schema, table)] = [(name, pg_type, map_pg_type(pg_type)) for name, pg_type in cur.fetchall()]

print(f"{len(selected)} table(s) to copy -> {TARGET_CATALOG}.{TARGET_SCHEMA}")
for _, s, t in selected:
    print(f"  {s}.{t} -> {TARGET_NAMES[(s, t)]} ({len(COLUMNS[(s, t)])} columns)")
if skipped:
    print(f"Skipped: {skipped}")

# COMMAND ----------

# DBTITLE 1,Copy
import time
from functools import reduce

if PARAMS["create_target_schema"] == "true":
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_PREFIX}")


def sql_string(text):
    """Escapes a Spark SQL string literal."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def pg_ident(name):
    return '"' + name.replace('"', '""') + '"'


def copy_table(schema, table):
    columns = COLUMNS[(schema, table)]
    ddl = spark_ddl([(name, spark_type) for name, _, spark_type in columns])
    converters = [make_converter(spark_type) for _, _, spark_type in columns]
    select_list = ", ".join(pg_ident(name) for name, _, _ in columns)

    frames, n_rows = [], 0
    # Named (server-side) cursor: rows are streamed batch by batch instead of loaded at once.
    with conn.cursor(name=f"copy_{uuid.uuid4().hex}") as cur:
        cur.itersize = FETCH_BATCH_ROWS
        cur.execute(f"SELECT {select_list} FROM {pg_ident(schema)}.{pg_ident(table)}")
        while True:
            batch = cur.fetchmany(FETCH_BATCH_ROWS)
            if not batch:
                break
            rows = [tuple(convert(value) for convert, value in zip(converters, row)) for row in batch]
            frames.append(spark.createDataFrame(rows, schema=ddl))
            n_rows += len(rows)

    df = reduce(lambda a, b: a.unionByName(b), frames) if frames else spark.createDataFrame([], schema=ddl)
    target = f"{TARGET_PREFIX}.{quote_ident(TARGET_NAMES[(schema, table)])}"
    # Creates the table when missing, otherwise replaces data and schema atomically (Delta history kept).
    df.writeTo(target).using("delta").createOrReplace()
    comment = (f"Copy of Lakebase {source_label} / {PARAMS['lakebase_database']} / {schema}.{table}, "
               f"refreshed {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC")
    spark.sql(f"COMMENT ON TABLE {target} IS '{sql_string(comment)}'")

    written = spark.table(target).count()
    if written != n_rows:
        raise RuntimeError(f"Row count mismatch: {n_rows} read, {written} written")
    return n_rows


results = []
try:
    for _, schema, table in selected:
        target_name = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.{TARGET_NAMES[(schema, table)]}"
        started = time.time()
        try:
            n_rows = copy_table(schema, table)
            results.append((f"{schema}.{table}", target_name, "OK", n_rows, round(time.time() - started, 1), None))
            print(f"  {schema}.{table}: {n_rows} row(s) -> {target_name}")
        except Exception as error:   # one failing table must not stop the others
            results.append((f"{schema}.{table}", target_name, "FAILED", None, round(time.time() - started, 1),
                            str(error)[:2000]))
            print(f"  {schema}.{table}: FAILED — {error}")
finally:
    conn.close()

# COMMAND ----------

# DBTITLE 1,Summary
summary = spark.createDataFrame(
    results, "source STRING, target STRING, status STRING, row_count BIGINT, seconds DOUBLE, error STRING")
display(summary)

failed = [r for r in results if r[2] == "FAILED"]
print(f"Done: {len(results) - len(failed)} table(s) copied, {len(failed)} failed, into {TARGET_CATALOG}.{TARGET_SCHEMA}")
if failed:
    raise RuntimeError(f"{len(failed)} table(s) failed: {[r[0] for r in failed]}")