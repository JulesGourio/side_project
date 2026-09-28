# Qualibot — evaluation and quality monitoring

Evaluation and monitoring of the Qualibot Knowledge Assistants on Databricks (MLflow GenAI).
Working conventions, environment and project status: see `CLAUDE.md`.

| Notebook | MLflow experiment | What it shows |
|---|---|---|
| `Build_Golden_Dataset.py` | evaluation experiment (Datasets tab) | the golden dataset `uat_landingzone.qualibot.qualibot_eval_golden` (20-30 reviewed cases) |
| `Evaluate_Knowledge_Assistant.py` | `.../qualibot-traces/trace_eval_all_v2` (traces in Unity Catalog) | one run per evaluation, one trace per case, 10 LLM judges and 3 code scorers registered, the golden dataset linked to every run |
| `Score_Production_QA.py` (job D_3) | `/Shared/qualibot-quality-scoring` | one run per scoring run, one trace per production turn, 8 LLM judges and 2 code scorers registered (about 6 judge calls per turn, about 3 turns per minute) |
| `Load_Test_Knowledge_Assistant.py` | `/Shared/qualibot-load-tests` | one run per load test: HTTP 429 and silent retrieval failures (answers without documents) per concurrency level |
| `Migrate_KA_Traces_To_UC.py` (job D_2) | `.../qualibot-traces/trace_ka_*` | the assistants' own traces, copied to Unity Catalog |

The scorers `relevance`, `language_match`, `groundedness`, `missed_answer` and `reference_integrity` are identical in
the evaluation and the production monitoring (shared cell of both notebooks), so their results can be compared.

## Evidence read by the retrieval judges

`groundedness`, `missed_answer` and `retrieval_sufficiency` read the `RETRIEVER` step `cited_document_excerpts`:

1. Documents: every document the answer relies on, i.e. the assistant's sources (production: `sources_json`;
   evaluation: the citations of the response) and the codes written in the answer, when they exist in the index.
   No limit on their number.
2. Searches, for each document: the question; each line of the answer that cites the document (`⟦n⟧` marker whose
   number is the document's `n` in `sources_json`, footnote pointing to it, or its code written in the line); each
   passage the assistant quoted from it (`#:~:text=` fragment of a citation link, footnote text).
3. Each search is a hybrid Vector Search query restricted to the document and its language variants; the 3 best
   chunks are kept (`EXCERPTS_PER_QUERY`), duplicates removed, untruncated.

The chunks come from the index `chunks_index_v1`: they are re-retrieved, not the passages the assistant itself read
(its traces do not expose them). A claim the chunks do not cover is "not verifiable", not "not supported".

## Judge model rate limits

The judge model (`databricks-gpt-5-6-luna`) allows 200,000 input and 20,000 output tokens per minute (1,000 requests
per second, 360,000 per hour). Both notebooks use 70% of the token limits (`JUDGE_RATE_SHARE`) and retry a rejected
call for about 2 minutes (`JUDGE_MAX_RETRIES`). The production scoring works in batches of about one minute of that
budget, writes each batch, and stops starting batches after `max_run_minutes`; the evaluation sets MLflow's scorer
rate limit (`MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT`) from the same budget.

## Deployment (Bitbucket Pipelines)

`bitbucket-pipelines.yml` (repository root of the bundle) holds manual custom pipelines. `deploy-uat-jobs` validates
and deploys the `qualibot-uat` target without `bundle run doc-compare`: jobs and notebooks are updated, the app is
neither redeployed nor restarted. It can be run on any branch, which deploys that branch's code to UAT.

## Dashboard data (Unity Catalog, `uat_proj.qualibot`)

The dashboard reads these tables directly (no view). Every table and column carries a Unity Catalog comment. Numeric
scores follow one convention everywhere: 1 = pass, 0 = fail, 0.5 = partial (`missed_answer` is reported as 1 = nothing
missed); labels have no numeric form. Evaluation rows carry their run's start time, subset and scorer configuration.

| Object | Grain | Written by | Typical use |
|---|---|---|---|
| `chat_quality_scores` | assistant turn | production scoring (D_3) | KPIs by day, assistant and division (`turn_verdict`, `groundedness_level`, `answer_type`, votes, cost); failure reasons (`failure_reasons`); review queue (`needs_human_review`, `golden_candidate`); turn drill-down |
| `chat_quality_assessments` | turn × scorer | production scoring | any scorer or label over time without schema change (`value`, `value_numeric`, `rationale`, `error`); shared scorers compared with `ka_eval_assessments` |
| `chat_quality_scoring_runs` | scoring run | production scoring | volumes, rates, estimated cost, judge/user agreement, turns left for the next run |
| `ka_eval_runs` | evaluation run | evaluation notebook | run context: endpoint, subset, judge model, scorer configuration |
| `ka_eval_metrics` | run × metric | evaluation notebook | score trends per assistant (same subset and configuration), with 95% confidence intervals |
| `ka_eval_results` | run × golden case | evaluation notebook | case drill-down and history (regressions, unstable cases): one column per metric, answer, documents, human rating |
| `ka_eval_assessments` | run × case × scorer | evaluation notebook | rationales, errors, human ratings |
| `ka_load_test_requests` | load test request | load test notebook | status, latency, documents, anomaly of each request |
| `ka_load_test_summary` | load test × level | load test notebook | 429, silent retrieval failures and latency per concurrency level |
| `ka_eval_golden_cases` | golden case | golden dataset builder | dataset composition and review progress |

Compare scores of the same `judge_config_id` (production) or `scorers_config_id` (evaluation): a change of judges,
judge model or verdict rules changes the fingerprint.

## Local tests

The notebooks cannot run outside Databricks; `tests/` runs their real cells locally with a real MLflow tracking store
(SQLite) and simulated judge model, Vector Search, assistant endpoint and Spark; the SQL they write (table DDL, inserts)
is replayed on a real local Spark session.

```bash
python -m venv .venv && .venv/bin/pip install "mlflow==3.11.1" pandas "sqlalchemy<2.0.40"   # minimum supported MLflow
.venv/bin/python tests/test_scoring.py        # production scoring: 4 scenarios, evidence searches, batches, tables, MLflow run
.venv/bin/python tests/test_scoring.py dry    # production scoring dry run
.venv/bin/python tests/test_eval.py 3         # evaluation on a 3-case sample; "" = full dataset
.venv/bin/python tests/test_refs.py           # document keys agree across the notebooks
.venv/bin/python tests/test_shared.py         # shared cell identical in both notebooks, answer cleaning, numeric scores
# Real Spark (needs Java): pyspark==3.5.3 and pandas in another environment
SQL_DUMP=/tmp/scoring.json .venv/bin/python tests/test_scoring.py && SQL_DUMP=/tmp/eval.json .venv/bin/python tests/test_eval.py ""
.spark/bin/python tests/test_sql.py /tmp/scoring.json /tmp/eval.json   # table DDL with comments, inserts, dashboard queries
.spark/bin/python tests/test_neighbours.py    # neighbour expansion of the golden dataset builder
.spark/bin/python tests/test_golden_export.py # flat golden cases table
.venv/bin/python tests/test_load.py          # load test against a simulated endpoint (needs requests, databricks-sdk)
```

Registering `@scorer` code scorers is only possible on Databricks: locally, the tests report them as "not registered".

## Data reference

Schemas and sample records of the source data, as observed in the UAT workspace.

### Document chunks — `uat_landingzone.qualibot.chunks_v1`

Source table of the Vector Search index `uat_landingzone.qualibot.chunks_index_v1`. One row per chunk.

| Column | Content |
|---|---|
| `IDDOC` | Numeric document id (e.g. `24924`) |
| `REF` | Document code (e.g. `GO-1536`) |
| `division` | `AS` or `IS` |
| `chunk_id` | Chunk id, `<IDDOC>-<type>-<n>` (e.g. `24924-IMG-003`) |
| `chunk_index` | Position of the chunk in the document (enables neighbour expansion) |
| `chunk_text` | Text, prefixed by a metadata header (see below) |
| `chunk_token_count` | Token count of the chunk |
| `chunk_content_type` | Type of content (e.g. `image`) |
| `semantic_headers` | JSON; for images: `image_label`, `volume_path`, `captions` |
| `chunk_sha256` | Hash of the chunk content |
| `url` | Intraqual link: `https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=<REF>` |
| `doc_date` | Publication date of the document |

Header of `chunk_text`: the title, division, category and date are in the text, not in dedicated columns.

```text
[Source: GO-1536 | Title: Guide outils pour préparateur de l'imprimante BRADY RFID | Division: AS |
 Category: AS - PROCESSES / PROCESSUS > R40- Produce | Date de diffusion: 2026-09-23 | Image: page ?, picture]
# [PHOTO_TECH] Poste de travail industriel équipé pour l'impression, le contrôle et la gestion informatique. ...
```

Sample `semantic_headers` of an image chunk:
`{"image_label": "picture", "volume_path": "/Volumes/uat_landingzone/qualibot/images/24924/24924_IMG_003.png", "captions": "[]"}`

### Chat logs — `uat_landingzone.qualibot.chat_messages`

One row per message (user or assistant).

| Column | Content |
|---|---|
| `id` | Message id |
| `session_id` | Conversation id |
| `role` | `user` / `assistant` |
| `content` | Message text (assistant answers cite documents with `⟦n⟧` markers or footnotes) |
| `created_at`, `deleted`, `deleted_at` | Timestamps and soft deletion |
| `division` | `ALL`, `AS` or `IS` |
| `endpoint_name` | Assistant endpoint (NULL on user messages) |
| `status`, `error_msg` | `ok` / error |
| `question_lang` | Language code of the question (e.g. `fr`, `cs`) |
| `reasoning_steps` | Reasoning summary of the assistant, concatenated |
| `sources_json` | Sources listed by the assistant: `[{"rank", "title", "url", "n"}]` |
| `tool_name`, `tool_query`, `tool_result` | Tool call fields (NULL in the samples) |
| `trace_id` | MLflow trace id of the assistant turn (e.g. `2793c134-0c43-427d-a74f-a6a8dd232ef5`) |
| `user_id`, `workspace_id`, `workspace_url` | Requester and workspace |

Sample `sources_json`:

```json
[{"rank": 0, "title": "P0043NF_CZ", "url": "https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=P0043NF_CZ", "n": 1},
 {"rank": 1, "title": "Q0025MI_GB", "url": "https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=Q0025MI_GB", "n": 2},
 {"rank": 2, "title": "P0043NF_MX", "url": "https://intraqual.lat.corp/intraqual_prod/doc/liredocumentdepuisrecherche?id=JIEvgxMhDmnHUWynY%2fRdQQ%3d%3d", "n": null}]
```

- `n` is the citation number in the answer; sources with `n = null` were returned but not cited.
- Some URLs carry no `?ref=` (`liredocumentdepuisrecherche?id=…`): the `title` is then the only document code.

### Knowledge Assistant response (trace output)

Streamed in the Responses format. The trace output is the list of stream events:

| Event `type` | Content |
|---|---|
| `response.reasoning_summary_text.delta` | Generic reasoning status lines ("Finalizing the set of top-ranked documents...") |
| `response.output_text.delta` | Answer text, token by token |
| `response.output_text.annotation.added` | One citation: `annotation.type = url_citation`, `title` = `url` = the Intraqual link with a `#:~:text=` fragment holding the cited passage |
| `response.output_item.done` | Final message: `item.content[].text` (full answer) and `custom_outputs.sources_used` |

Final answer: documents cited in bold (`**Q0196QP_FR**`), footnotes `[^eOF0-n]` quoting the cited passage and its
link, and a "Sources" table (REF, title, version, division, date).

Condensed example:

```yaml
- type: response.output_text.annotation.added
  annotation:
    type: url_citation
    title: https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=Q0196QP_FR
    url: https://intraqual.lat.corp/intraqual_prod/identification.aspx?ref=Q0196QP_FR#:~:text=V%C3%A9rification%20de%20la%20connaissance...
- type: response.output_item.done
  custom_outputs: {sources_used: true}
  item: {type: message, role: assistant, content: [{type: output_text, text: "Pour suivre les compétences des opérateurs, ..."}]}
```

### Observations relevant to the notebooks

- `chunk_index` exists: excerpts can be expanded to neighbouring chunks of the same document.
- Title, category and date are only available inside the `chunk_text` header (parse it for a document catalogue).
- The trace output shows no retrieval step with chunk texts; the only retrieved text is the cited passage in each
  citation (`#:~:text=` fragment and footnotes). Whether the trace spans hold more remains to be checked.
- Document codes appear with a dot before the language suffix (`PRLAT549.FR`) and with suffixes such as `_BG`:
  both must be handled by the document key (`LANG_SUFFIXES` and separators).
