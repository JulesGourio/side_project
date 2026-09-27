# Qualibot — working conventions for Claude Code

## Writing rules (all files)
- Code, comments, notebook markdown, printed messages and YAML comments: professional English.
- Every file is standalone: never reference a previous version, a change history or a version label (v1, v2…)
  in contents or file names. Resource names that contain a version (e.g. the agent `qualibot_ALL_v2`) are kept as is.
- Replies to the user: in French, starting with a one- or two-sentence summary of what is being done.

## Working mode with Claude Code
- Claude Code has NO access to Databricks. The user runs the notebooks and jobs, then sends back outputs, table
  extracts (CSV / JSON / `DESCRIBE TABLE`) or screenshots. Never claim something ran in Databricks; state what was
  only tested locally with stubs.
- Target quality bar: everything visible and linked in MLflow — Runs, Traces, Datasets (golden dataset linked to
  the runs), Judges / Scorers (every scorer registered), clean professional notebooks (no draft cells, no dead code).
- Schemas and samples of the source data sent by the user (chunks table, chat logs, assistant trace output) are
  recorded in `README.md`, section "Data reference". Read it before touching data access code.
- Replies: first one or two sentences saying what is being done, then details. Short, no unnecessary narration.
- This repository is an extract of the bundle repository: the files are flat at its root; the bundle paths below
  (`utils/...`, `resources/...`) are where they live in the deployed Databricks bundle.

## Repository layout (evaluation and quality)
| Path | Purpose |
|---|---|
| `utils/evaluation/Build_Golden_Dataset.py` | Builds the golden evaluation dataset (cached in `uat_landingzone.qualibot.qualibot_eval_cache`) and exports it to the MLflow dataset `uat_landingzone.qualibot.qualibot_eval_golden` |
| `utils/evaluation/Evaluate_Knowledge_Assistant.py` | Evaluates a Knowledge Assistant endpoint on the golden dataset with MLflow GenAI (traces, judges, report) |
| `utils/quality_monitoring/Score_Production_QA.py` + `resources/quality_scoring.yml` | Twice-daily LLM-judge scoring of production turns (job `D_3_qualibot-quality-scoring`) |
| `utils/evaluation/Load_Test_Knowledge_Assistant.py` | Load test of an assistant endpoint: HTTP 429 and silent retrieval failures per concurrency level (tables `ka_load_test_*`) |
| `utils/traces_migration/Migrate_KA_Traces_To_UC.py` + `resources/traces_migration.yml` | Nightly copy of the assistants' MLflow traces to Unity Catalog (job `D_2_qualibot-traces-sync`) |
| `tests/` (this repository only) | Local end-to-end tests of the notebooks: real MLflow (SQLite), simulated judge model, Vector Search, assistant and Spark (see `README.md`, "Local tests") |

## MLflow design (evaluation and monitoring)
- Every judge is an MLflow scorer registered in its experiment (Judges / Scorers tab), never scheduled (no background
  cost): `make_judge` judges, built-in judges, and `@scorer` code scorers.
- Both notebooks contain the same "Shared scorers and helpers" cell (`tests/test_shared.py` checks it is identical):
  judges `relevance`, `language_match`, `groundedness`, `missed_answer`, code scorer `reference_integrity`, answer
  cleaning (citation `#:~:text=` fragments removed before judging), numeric form of verdicts, scorer registration and
  Unity Catalog write helpers. Edit it in one notebook and copy it to the other.
- The retrieval judges `groundedness` and `missed_answer` read the excerpts of the cited documents from the trace's
  `RETRIEVER` step (`cited_document_excerpts`) and return no assessment when there is none. The excerpts are a subset of
  the documents: a claim absent from them is "not verifiable", not "not supported".
- Judges are skipped when not applicable: `user_reaction` only when the user wrote again (otherwise `no_next_turn`,
  free), retrieval judges only when the answer cites indexed documents, `safety` on a stable 10% sample of the turns;
  `answer_type` also carries the completeness (`answered_full` / `answered_partial`). Scorers are registered again only when their
  fingerprint changes (experiment tag `qualibot.scorers_config_id`).
- Dashboard data: documented Unity Catalog tables and views in `uat_proj.qualibot` (catalogue in `README.md`,
  "Dashboard data"); rows are replaced by key (`message_id` in production, `run_id` in evaluation).
- Production monitoring replays the stored answers through `mlflow.genai.evaluate` (`replay_turn`), so each scored turn
  is a trace with its conversation, answer, excerpts, every verdict, the rule-based `turn_verdict` and the user's vote.
- Evaluation runs on the full golden dataset are linked to it by `evaluate`; sampled runs are linked with `log_input`.

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
- MLflow minimum 3.11: `make_judge(feedback_value_type=...)` exists from 3.6, and before 3.11 a `databricks:/<endpoint>`
  judge model requires LiteLLM (the judge check then silently falls back to the Databricks-managed judge).
- A registered `@scorer` stores only its function body: it must be self-contained (imports inside, no notebook
  globals; the judge model is read from the trace tag `judge_model`). Registering `@scorer` functions only works on a
  Databricks tracking server.
- `mlflow.genai.evaluate` needs a `predict_fn` decorated with `@mlflow.trace`, otherwise its spans end up in separate
  traces; `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=True` avoids one extra call before the run.
- Built-in retrieval judges raise an error on a trace without a `RETRIEVER` span and judge an empty span as
  unsupported: wrap them in a `@scorer` that returns None when there are no excerpts.
- Assessments in error have a None value: never map it to a score (it is not "none").
- Reading traces stored in Unity Catalog requires `MLFLOW_TRACING_SQL_WAREHOUSE_ID`.
- An MLflow experiment's parent folder must exist (`w.workspace.mkdirs`).
- Document codes: compare with a key insensitive to language suffix (`_FR`, `.FR`, `_GB`, `_BG`…), separators, case
  and zero padding (`IN_APO_006` = `IN_APO_0006`, a typo present inside some documents). Codes ending with letters
  (`Q0062MI`, `H0049MR`) are valid codes. The same key is used in the three notebooks and in `document_recall`
  (`tests/test_refs.py` checks they agree).
- A cited code absent from the index is not necessarily invented: documents often reference procedures outside the
  corpus. Only codes found neither in the index nor in the cited excerpts are "unverified".

- Deployment: `bitbucket-pipelines.yml`; the custom pipeline `deploy-uat-jobs` deploys the UAT bundle (jobs,
  notebooks) without redeploying the app.

## Checks before handing over a change
- `python -m py_compile` on every modified notebook; `databricks bundle validate -t qualibot-uat`.
- Run the local tests (`README.md`, "Local tests") on the minimum MLflow version (3.11) and on the latest one, and
  replay the generated SQL on a real Spark session (`tests/test_sql.py`).

## Project status

### Delivered
| Component | State |
|---|---|
| Trace migration (`D_2_qualibot-traces-sync`) | Deployed in UAT. Manual runs: `trace_test`, then `trace_ka_all_v2`, then `trace_ka_is_v2,trace_ka_as_v2`. Schedule to unpause with `to_migrate: "*"` once validated. |
| Production scoring (`D_3_qualibot-quality-scoring`) | Registered MLflow scorers (about 6 judge calls per turn), tables `chat_quality_*` and 5 views; failure e-mail configured, quality alerts e-mailed only once `fail_on_alert` is "true" ("false" for now); tested end to end locally, not yet run in Databricks. Next run: `dry_run=true`, then `reset_outputs=true` with `test_limit=20`, then a full run, then unpause the schedule. |
| Evaluation notebook | Shared scorers with monitoring, every scorer registered, dataset linked to every run, tables `ka_eval_*` and 3 views; tested end to end locally, not yet run in Databricks. |
| Golden dataset builder | 20-30 cases, compliance-matrix quota, neighbour expansion by `chunk_index`, flat table `ka_eval_golden_cases`; reuses the existing cache; not yet run in Databricks. |

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
- Is there a document metadata table (title, language, status current/obsolete), e.g. from the parsing pipeline? Titles
  are otherwise only in the `chunk_text` header of `chunks_v1`.
- Do the assistants' trace spans hold the retrieved chunk texts? The trace output only carries the cited passages
  (`#:~:text=` fragments); if spans hold more, the evaluation can measure the assistant's actual retrieval.
- Errors of the D_2 and D_3 runs (the user will send them).
- Knowledge Assistant experiments may offer a native "Delta sync" trace archival option; if available, it could replace
  the nightly migration job for new traces.

### Next tasks, in priority order
1. Run the production scoring (see Delivered) and check the Traces, Judges and Runs tabs of
   `/Shared/qualibot-quality-scoring` and the tables and views of `uat_proj.qualibot`; build the dashboard on the views
   (`README.md`, "Dashboard data").
2. Run `Build_Golden_Dataset.py` with `FORCE = {"reformulations", "evidence_pool", "ka_fresh"}` to benefit from the
   retrieval routes (hypothetical answer, current assistant sources, similar and adjacent chunks), review in section 11,
   export 20-30 cases.
3. Run `Evaluate_Knowledge_Assistant.py` (`sample_n=5`, then full), rate 10+ answers in "Human labels", check agreement,
   and align `fact_coverage` once 10+ ratings exist.
4. Improve the assistant's retrieval of acronyms and document types (expanded acronyms and full titles in chunk text,
   through the parsing pipeline) and its instructions (cite only exact codes, never assert compliance or rules that are
   not written); measure each change with the evaluation notebook on the same subset.
5. Grow the golden dataset with validated production failures and compliance-matrix questions (within 20-30 cases).
6. Once stable in UAT, add `qualibot-prod` targets (YAML anchors, as in the parsing pipeline) with prod ids and catalogs.
