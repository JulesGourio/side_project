# Qualibot — working conventions for Claude Code

## Writing rules (all files)
- Code, comments, notebook markdown, printed messages and YAML comments: professional English.
- Every file is standalone: never reference a previous version, a change history or a version label (v1, v2…)
  in contents or file names. Resource names that contain a version (e.g. the agent `qualibot_ALL_v2`) are kept as is.
- Replies to the user: in French, starting with a one- or two-sentence summary of what is being done.

## Repository layout (evaluation and quality)
| Path | Purpose |
|---|---|
| `utils/evaluation/Build_Golden_Dataset.py` | Builds the golden evaluation dataset (cached in `uat_landingzone.qualibot.qualibot_eval_cache`) and exports it to the MLflow dataset `uat_landingzone.qualibot.qualibot_eval_golden` |
| `utils/evaluation/Evaluate_Knowledge_Assistant.py` | Evaluates a Knowledge Assistant endpoint on the golden dataset with MLflow GenAI (traces, judges, report) |
| `utils/quality_monitoring/Score_Production_QA.py` + `resources/quality_scoring.yml` | Twice-daily LLM-judge scoring of production turns (job `D_3_qualibot-quality-scoring`) |
| `utils/traces_migration/Migrate_KA_Traces_To_UC.py` + `resources/traces_migration.yml` | Nightly copy of the assistants' MLflow traces to Unity Catalog (job `D_2_qualibot-traces-sync`) |

## Environment (UAT workspace)
- Chat logs: `uat_landingzone.qualibot.chat_messages` (has `trace_id`, `sources_json`), `chat_feedbacks`.
- Document index: `uat_landingzone.qualibot.chunks_index_v1` (endpoint `qualibot`, columns `REF`, `chunk_text`, `semantic_headers`).
- Outputs: `uat_proj.qualibot` (scores, scoring runs, trace tables `trace_*`).
- Assistants: ALL `ka-7679a56e-endpoint` (the one evaluated), IS `ka-1560aded-endpoint`, AS `ka-3a7e9255-endpoint`
  (IS and AS are subsets of ALL by department). Max 3 concurrent calls per endpoint.
- Judge model: `databricks-gpt-5-6-luna` (2.857 DBU/M input tokens, 17.143 DBU/M output tokens).
- SQL warehouse for Unity Catalog traces: `5890912c31867b77`.
- Job identity (UAT): service principal `3e5cd4e5-f765-4760-b974-ee7715258b39`; it owns the output tables.

## Bundle conventions
- Resources are defined per target (`qualibot-uat`, `qualibot-prod`) with `run_as` the target's service principal.
- Job names follow `D_<n>_<name>-${bundle.target}`; tags `project`, `activity-type`, `job-purpose`; CoreAdmin and
  CoreDev groups get CAN_MANAGE.
- Classic single-node clusters: `SINGLE_USER`, `ON_DEMAND`, no `policy_id` (org policies force spot).
- Notebooks are deployed with `source: WORKSPACE` and paths relative to the resource file.

## Technical rules learned the hard way
- Serverless: never `pip install -U` blindly. Install only missing packages with the already installed ones pinned
  (see the setup cell of each notebook); a protobuf downgrade prevents the kernel from starting.
- `from IPython.display import display` must be aliased (`ipy_display`): it otherwise hides Databricks' `display`.
- MLflow `Correctness` accepts `expected_facts` OR `expected_response`, never both.
- Reading traces stored in Unity Catalog requires `MLFLOW_TRACING_SQL_WAREHOUSE_ID`.
- An MLflow experiment's parent folder must exist (`w.workspace.mkdirs`).
- Document codes: compare with a key insensitive to language suffix (`_FR`, `_GB`, `_EN`…), separators, case and
  zero padding (`IN_APO_006` = `IN_APO_0006`, a typo present inside some documents). Codes ending with letters
  (`Q0062MI`, `H0049MR`) are valid codes.
- A cited code absent from the index is not necessarily invented: documents often reference procedures outside the
  corpus. Only codes found neither in the index nor in the cited excerpts are "unverified".

## Checks before handing over a change
- `python -m py_compile` on every modified notebook; `databricks bundle validate -t qualibot-uat`.
- Pure functions are tested locally with stubs for Spark / MLflow / Databricks SDK when possible.
