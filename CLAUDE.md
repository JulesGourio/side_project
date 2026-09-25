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

## Project status

### Delivered
| Component | State |
|---|---|
| Trace migration (`D_2_qualibot-traces-sync`) | Deployed in UAT. Manual runs: `trace_test`, then `trace_ka_all_v2`, then `trace_ka_is_v2,trace_ka_as_v2`. Schedule to unpause with `to_migrate: "*"` once validated. |
| Production scoring (`D_3_qualibot-quality-scoring`) | Deployed in UAT, validated on 20 turns. Next run: `reset_outputs=true`, `test_limit=20`, then a full run, then unpause the schedule. |
| Evaluation notebook | Written, pure functions tested locally; not yet run end to end in Databricks. |
| Golden dataset builder | Written, reuses the existing cache; not yet run in Databricks. |

### Findings from the first evaluation run (25 cases, qualibot_ALL_v2)
- The raw correctness score (44%) underestimated the assistant: about a third of the failures came from the
  measurement or the dataset; the corrected estimate is around 50-55%.
- Strengths: refusals and out-of-scope requests, questions answered by a single well-indexed document.
- Main weakness, retrieval: documents searched through an acronym or a document type are not found
  (APO → glossary IN_APO_0006, CMP → template NF-10065, SAP movement → MI-13841_GB).
- Generation: occasional inferences not written in the documents (FAI after a site change), and a cited code typo
  (IN_APO_006) that also exists in the documents themselves.
- Production traffic: about half of the questions come from one use case, filling customer compliance matrices
  (e.g. Dassault requirements: "is Latécoère compliant, which document proves it?"). Its bad-answer rate is high, and
  the assistant tends to assert compliance instead of citing the documents that address the requirement.

### Open questions (to ask the user or an expert)
- Expert: unspecified bore tolerance on Airbus drawings — NSA2010 / ABS1707 (golden) or NSA2110 (assistant)?
- Expert: margin rate applied in inter-site invoicing (P&L LEAP case).
- Schema of the index source table: is there a chunk order column (neighbour expansion), a title, a status, a division?
- Is there a document metadata table (title, language, status, division), e.g. from the parsing pipeline?
- Do the assistants' traces contain a retrieval step with the retrieved chunk texts? If so, the evaluation can measure
  the assistant's actual retrieval instead of the excerpts of the cited documents.
- Target size of the golden dataset (25 today; 60-100 with validated production failures) and whether a dedicated set
  of compliance-matrix questions is wanted.
- Knowledge Assistant experiments may offer a native "Delta sync" trace archival option; if available, it could replace
  the nightly migration job for new traces.

### Next tasks, in priority order
1. Run the production scoring end to end (see Delivered) and check the Traces tab of `/Shared/qualibot-quality-scoring`.
2. Run `Build_Golden_Dataset.py` with `FORCE = {"reformulations", "evidence_pool", "ka_fresh"}` to benefit from the
   new retrieval routes (hypothetical answer, current assistant sources, expansion), review in section 11, export.
3. Run `Evaluate_Knowledge_Assistant.py` (`sample_n=5`, then full), rate 10+ answers in "Human labels", check agreement,
   and align `fact_coverage` once 10+ ratings exist.
4. Improve the assistant's retrieval of acronyms and document types (expanded acronyms and full titles in chunk text,
   through the parsing pipeline) and its instructions (cite only exact codes, never assert compliance or rules that are
   not written); measure each change with the evaluation notebook on the same subset.
5. Grow the golden dataset with validated production failures and compliance-matrix questions.
6. Once stable in UAT, add `qualibot-prod` targets (YAML anchors, as in the parsing pipeline) with prod ids and catalogs.
