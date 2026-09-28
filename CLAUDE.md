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
| `utils/evaluation/Load_Test_Knowledge_Assistant.py` | Load test of an assistant endpoint: one MLflow trace per request traced from the caller's side (429 in state ERROR, the assistant's own steps copied, empty retrieval marked ERROR), one child run per concurrency level, charts; tables `ka_load_test_*` |
| `utils/traces_migration/Migrate_KA_Traces_To_UC.py` + `resources/traces_migration.yml` | Nightly copy of the assistants' MLflow traces to Unity Catalog (job `D_2_qualibot-traces-sync`) |
| `tests/` (this repository only) | Local end-to-end tests of the notebooks: real MLflow (SQLite), simulated judge model, Vector Search, assistant and Spark (see `README.md`, "Local tests") |

## MLflow design (evaluation and monitoring)
- Every judge is an MLflow scorer registered in its experiment (Judges / Scorers tab), never scheduled (no background
  cost): `make_judge` judges, built-in judges, and `@scorer` code scorers.
- Both notebooks contain the same "Shared scorers and helpers" cell (`tests/test_shared.py` checks it is identical):
  judges `relevance`, `language_match`, `groundedness`, `missed_answer`, code scorer `reference_integrity`, answer
  cleaning (citation `#:~:text=` fragments removed before judging), numeric form of verdicts, scorer registration and
  Unity Catalog write helpers. Edit it in one notebook and copy it to the other.
- The assistants query the `qualibot` index (`chunks_index_v1`). The judges read what the assistant retrieved: the
  passages of the RETRIEVER spans of its own MLflow trace, copied into the scoring trace as the RETRIEVER step
  `assistant_retrieval` (production: `mlflow.get_trace` on the `trace_id` of `chat_messages`, as is then as
  `tr-<32 hex>`; evaluation: `databricks_options.return_trace`). Only when a trace shows no retrieval step do they read
  `cited_document_excerpts` (re-retrieved excerpts of the cited documents, a subset: absent = "not verifiable").
  `answer_context.evidence_source` says which; each trace has exactly one RETRIEVER step. The production notebook's
  "Assistant traces" cell prints the span structure of the assistant's trace when no retrieval step is recognised.
- `retrieval_quality` (shared) separates retrieval from generation errors: a judge on the assistant's passages, then,
  when they are not sufficient, the same judge on `corpus_search` (independent search of the whole index, TOOL step)
  → `sufficient` / `retrieval_miss` / `documentation_gap` / `not_applicable`. The turn verdict maps each failure reason
  to a stage (`REASON_STAGE`) and writes `error_source`. `compliance_claim` (shared) targets compliance-matrix answers
  asserting compliance beyond the evidence.
- Never cap what the judges read (user requirement): all retrieved passages or every document the answer relies on,
  untruncated chunks, the whole conversation window the assistant saw.
- Judges are skipped when not applicable: `user_reaction` only when the user wrote again (otherwise `no_next_turn`,
  free), retrieval judges only when the answer cites indexed documents, `safety` on a stable 10% sample of the turns;
  `answer_type` also carries the completeness (`answered_full` / `answered_partial`). Scorers are registered again only when their
  fingerprint changes (experiment tag `qualibot.scorers_config_id`).
- Dashboard data: documented Unity Catalog tables in `uat_proj.qualibot`, no view (user decision; catalogue in
  `README.md`, "Dashboard data"); rows are replaced by key (`message_id` in production, `run_id` in evaluation);
  evaluation rows carry their run context (start time, subset, scorer configuration) so no join is needed.
- Judge pacing: `judge_rate_share` (widget, 0.7 by default) of the judge model's limits (200k input / 20k output tokens per minute),
  measured with the tokens the judge model reports in each assessment's metadata (`mlflow.assessment.judgeInputTokens`
  / `judgeOutputTokens`, also used for costs and written in the tables). Production scores batches of about one minute of that budget, writes
  each batch, stops starting batches after `max_run_minutes` (turns left go to the next run) and slows down after a
  rate-limit error; evaluation sets `MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT`. `MLFLOW_GENAI_EVAL_MAX_RETRIES=7` in both
  (MLflow's own 429 backoff: 1 s … 60 s). Each production batch logs one `dataset` input to the run (MLflow behaviour).
  A run stops if the judge endpoint does not answer (no silent fallback to another judge model).
- Production monitoring replays the stored answers through `mlflow.genai.evaluate` (`replay_turn`), so each scored turn
  is a trace with its conversation, answer, excerpts, every verdict, the rule-based `turn_verdict` and the user's vote.
- Evaluation runs on the full golden dataset are linked to it by `evaluate`; sampled runs are linked with `log_input`.

## Environment (UAT workspace)
- Chat logs: `uat_landingzone.qualibot.chat_messages` (has `trace_id`, `sources_json`), `chat_feedbacks`.
- Document index: `uat_landingzone.qualibot.chunks_index_v1` (endpoint `qualibot`, columns `REF`, `chunk_text`, `semantic_headers`).
- Outputs: `uat_proj.qualibot` (scores, scoring runs, trace tables `trace_*`).
- Assistants: ALL `ka-7679a56e-endpoint` (the one evaluated), IS `ka-1560aded-endpoint`, AS `ka-3a7e9255-endpoint`
  (IS and AS are subsets of ALL by department). Max 3 concurrent calls per endpoint.
- Judge model: `databricks-gpt-6-luna` (user choice): 1.4 DBU/M input and 7.1 DBU/M output tokens; limits 200k input
  and 20k output tokens per minute, 1,000 requests per second, 360k per hour.
- SQL warehouse for Unity Catalog traces: `5890912c31867b77`.
- Job identity (UAT): service principal `3e5cd4e5-f765-4760-b974-ee7715258b39`; it owns the output tables.

## Bundle conventions
- Resources are defined per target (`qualibot-uat`, `qualibot-prod`) with `run_as` the target's service principal.
- Job names follow `D_<n>_<name>-${bundle.target}`; tags `project`, `activity-type`, `job-purpose`; CoreAdmin and
  CoreDev groups get CAN_MANAGE.
- Classic single-node clusters: `SINGLE_USER`, single-node `spark_conf` and `ResourceClass` tag. The "Job Compute" policies
  (`1. Job Compute XS`: single node, `.large` nodes, `auto:latest-lts`) set `SPOT_WITH_FALLBACK` with `first_on_demand: 1`,
  so a single node (the driver) stays on demand. The scoring job uses `1. Job Compute XS` through a bundle variable
  lookup (`job_compute_xs_policy_id`); the job identity needs CAN_USE on the policy.
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
- Never put the user's next message in the evaluation inputs: the judges read it as the question (every answer then
  looks off-topic). It is read only by `user_reaction`. The built-in `Safety` judge takes `outputs` only.
- Trace tags hold short identifiers only: on Databricks a tag value over ~250 characters makes the whole trace fail
  (turn not scored). Variable-length data (document lists, next user message, errors) goes to the trace step
  `answer_context` (`record_answer_context`), which the scorers read.
- `relevance` and `answer_type` must not judge language, facts or evidence.
- `databricks-gpt-6-luna` does not support batch inference (`ai_query` fails with PERMISSION_DENIED): the golden
  dataset builder calls the endpoint directly from the driver (`llm()`: one call per distinct prompt, `LLM_WORKERS` in
  flight, paced on `LLM_RATE_SHARE` of the token limits, 429 retried) and writes each step to the cache by parts of
  `LLM_KEYS_PER_WRITE` rows, so an interrupted step resumes. Embeddings still use `ai_query`.
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
| Production scoring (`D_3_qualibot-quality-scoring`) | Registered MLflow scorers (about 8 judge calls per turn), tables `chat_quality_*` (no view); failure e-mail configured, quality alerts e-mailed only once `fail_on_alert` is "true" ("false" for now); tested end to end locally, not yet run in Databricks. Test run on 20 turns done in UAT; `reset_outputs=true` replaces the tables only once the new scores are written. Three 20-turn runs reviewed with the user; the third one is validated (20/20 scored, no scorer error, every bad verdict traced to a real assistant error). Judges now read the passages the assistant retrieved (from its trace), `retrieval_quality` and `compliance_claim` added, `error_source` (retrieval / generation), measured judge tokens and costs, Luna 6, batches paced on the judge limits and written one by one, time budget `max_run_minutes`, legacy views dropped by the job: tested locally only. First UAT run with the assistant's passages (20 turns): every assistant trace resolved (chat_messages trace_id = MLflow `tr-…` id), 2 retrieval steps and 10-14 passages per turn, evidence from the assistant's retrieval for 20/20, no scorer error; 1 good / 10 acceptable / 9 bad, error_source generation 19/20, retrieval_quality sufficient 10, documentation_gap 8, retrieval_miss 1; about 60k input and 2.7k output judge tokens per turn (8.8 calls), $0.007 per turn, 20 turns in 9 minutes (input limit binding: about 2.3 turns per minute, about 225 turns per 100-minute run). Verdicts reviewed: the 9 bad are real assistant errors (7 compliance answers asserting "Oui, Latécoère répond" beyond the evidence, GO-1508 said to make the inspector close the NCR, P0289MI given instead of P0053MI); most acceptable verdicts came from status labels ("Courant") of the closing sources table, now ignored by `groundedness` unless the evidence shows the document cancelled or replaced. `retrieval_quality` judged a rework request ("remove Q0451MQ") as not applicable: it now judges the question the previous answer addressed, and the independent search uses the last two user messages. A retrieval miss now always counts in `error_source`. Re-run on 20 turns with these refinements: 2 good / 12 acceptable / 6 bad, answer types answered_full 11, answered_partial 6, not_found 2, clarification_request 1; retrieval_quality documentation_gap 10, sufficient 7, retrieval_miss 2, not_applicable 1; error_source generation 16, retrieval 2; 8.6 judge calls and $0.006 per turn, 20 turns in 7 minutes. Acceptable verdicts reviewed: mostly `partially_supported` on paraphrases or abbreviated titles (now: judge meaning, not wording; lower only for a claim that matters — value, rule, role, scope, condition, document identity); a clarification request to the one-word question "test" was a false `missed_answer_in_sources` (a genuinely ambiguous request is no longer a miss). Bad verdicts confirmed (APO glossary not retrieved, English versions not retrieved, GO-1508 misread, test-strap department not in the documents, P0289MI instead of P0053MI). The job runs on a classic single-node job cluster with the policy `1. Job Compute XS` (mlflow installed as a job library). First run on this cluster: 30 turns (the 30 most recent: a `test_limit` was still set), 13 minutes, no error, the assistant's retrieval found for 30/30, one judge configuration; AS assistant (`ka-3a7e9255`) bad rate 69% (9 unsupported compliance claims out of 13 turns), ALL assistant 24%; `compliance_claim` never `evidence_based` when applicable; no user vote among these turns. `n_turns_left` now counts every turn still waiting (it only counted the turns loaded by the run). MLflow run metrics `<scorer>/mean` are logged over the whole run at the end (mlflow.genai.evaluate logs them per batch, so its values covered the last batch only); the notebook header documents every run metric and the verdict rules. Job defaults: `max_run_minutes=330` (task timeout 6 h; a scheduled run ends in minutes), widget `judge_rate_share` (0.7; 0.9 for a backlog). Next: score the backlog with `test_limit` empty (`reset_outputs=false`, `rescore_changed_config=true`, `max_run_minutes=100`), then unpause the schedule. |
| Evaluation notebook | Shared scorers with monitoring, every scorer registered, dataset linked to every run, tables `ka_eval_*` (no view); tested end to end locally, not yet run in Databricks. |
| Golden dataset builder | 20-30 cases, compliance-matrix quota, neighbour expansion by `chunk_index`, flat table `ka_eval_golden_cases`; reuses the existing cache; not yet run in Databricks. Judge model called directly (no batch inference for Luna 6), steps written by parts. `ka_fresh` asks the assistant for its trace: the documents it retrieved feed the evidence pool (origin `assistant_retrieval`) and give `ka_failure_stage` (retrieval: none of the expected documents retrieved / generation / unknown) for the cases it fails. Production failures are selected by stage: slots `production_retrieval_miss` (3), `production_compliance_claim` (3), `production_failure` (4), from `chat_quality_scores`. No cap on the generators' context (every chunk graded 2 or 3) and no truncation of messages or answers in the prompts. Compliance questions: generation rule and exported guideline `COMPLIANCE_GUIDELINE` (cite the documents addressing the requirement, never assert compliance beyond them). First run with these changes: the assistant returned its trace for 36/36 shortlisted cases; final selection 25 cases (19 log, 1 override, 5 synthetic), assistant verdicts partially_correct 16, incorrect 6, correct 1, unjustified_refusal 1, justified_refusal 1; answerability partial 12, full 7, none 4, out_of_scope 2. Too few cases the assistant passes (MIN_KA_OK = 8): slot `production_pass` (6) added, from turns the production scoring judged good and nobody voted down; the selection now warns when a minimum is not reached. Next: after the D_3 backlog, rerun the builder with `FORCE = set()` (the shortlist is rebuilt at every run, new cases go through the cached steps), review in section 11, export. |

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

### Findings from the validated production scoring run (20 recent turns, mostly ALL)
- 30% bad, 40% acceptable, 30% good. Every bad verdict comes from claims contradicted by the cited documents:
  - compliance-matrix answers ("idem pour <Dassault requirement>") assert "Oui, Latécoère répond à cette exigence" and
    cite documents that do not say it (Q0070MI, Q0406MI, MI-1226 instead of Q0258MM, a Quality Manual clause about
    authorities presented as a Dassault right, a wrong title for Q0408QP);
  - SAP procedural details invented (COHV and Winshuttle mass treatment, exception message numbers confused with user
    statuses) and a cancelled instruction (IAQ 04 21 01) presented as current.
- Frequent warnings: answers that leave out details the cited documents contain (repair case of NCR closure, LBG listed
  among the applicable sites), and an English answer to a Bulgarian question.

### Findings from the load tests (one question, ALL assistants)
- `ka-7679a56e-endpoint`: HTTP 429 from 10 requests in flight (23/60 at 10, 29/60 at 30), no `Retry-After` header;
  silent retrieval failures (HTTP 200, empty retrieval step) from 10 in flight (1/60 at 10, 7/60 at 30).
- `ka-112b2b12-endpoint`, 40 simultaneous requests: 8/40 answers without any document; their trace shows an empty
  `docs` RETRIEVER step, no `rerank` step, and every step logged `OK`; some retrieval steps carry the exception
  "The given endpoint does not exist, please retry…" (the assistant's internal Vector Search / reranker call rejected).
- The assistant's own experiment held 31 traces for 40 requests, all in state OK: rejected requests leave no trace
  there and silent failures are not flagged. Hence the load test traces every request from the caller's side.

### Open questions (to ask the user or an expert)
- Expert: unspecified bore tolerance on Airbus drawings — NSA2010 / ABS1707 (golden) or NSA2110 (assistant)?
- Expert: margin rate applied in inter-site invoicing (P&L LEAP case).
- Is there a document metadata table (title, language, status current/obsolete), e.g. from the parsing pipeline? Titles
  are otherwise only in the `chunk_text` header of `chunks_v1`.
- Errors of the D_2 and D_3 runs (the user will send them).
- Knowledge Assistant experiments may offer a native "Delta sync" trace archival option; if available, it could replace
  the nightly migration job for new traces.

### Next tasks, in priority order
1. Run the production scoring (see Delivered) and check the Traces, Judges and Runs tabs of
   `/Shared/qualibot-quality-scoring` and the tables of `uat_proj.qualibot`; build the dashboard on the tables
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
6. Judge refinements to decide on the first runs with the assistant's passages: claim-by-claim verification of the
   answer (one verdict and quoted passage per claim) if `groundedness` rationales stay too coarse; a second
   independent opinion on bad verdicts if false bad verdicts appear (none in the validated run).
7. Once stable in UAT, add `qualibot-prod` targets (YAML anchors, as in the parsing pipeline) with prod ids and catalogs.
