# LatLang

Bilingual `.docx` translation pipeline for Latecoere aerospace documentation,
running as its own standalone Databricks App. See
`client/src/components/translate/README.md` for the full pipeline
documentation (extract → audit → Q&A → translate → fit_check → rebuild →
validate → preview) and the Lakebase schema.

## Independence

Fully standalone app: own FastAPI backend, own React frontend, own Lakebase
database (`latlang`/`latlang_test`), own catalog schema/volume, own
Databricks App and service principal. `LAKEBASE_PROJECT_ID` reuses an
existing shared Lakebase Postgres project id to avoid provisioning new
compute — that's the only shared infra, and this app owns its own database
inside it. No runtime dependency on any other app.

## Local development

```bash
uv pip install -e .
cd client && bun install && bun run dev   # or npm install / npm run dev
```

Backend: `uvicorn server.app:app --reload`

## Deployment

Databricks Asset Bundle:

```bash
databricks bundle validate -t latlang-uat
databricks bundle plan -t latlang-uat -o json
databricks bundle deploy -t latlang-uat
```

Never `deploy`/`destroy` on `latlang-uat` or `latlang-prod` without explicit
approval — `latlang-uat-test` is the disposable validation target.

## Open items before first real deploy

- Create the `Role-Project-LEAP-End-users-LatLang` Databricks group (or agree
  on its final name) and grant it access in `databricks.yml`.
- Confirm `TRANSLATE_ENDPOINT` / `GLOSSARY_EXTRACT_ENDPOINT` Model Serving
  endpoint names for this app's own usage.
- Build and upload this app's own LibreOffice archive
  (`utils/soffice_packaging/package_libreoffice.py`) to the dedicated UC
  volume, then set `SOFFICE_ARCHIVE_VOLUME_PATH`.
- Run `migrate_translation_data_uat_test` (see `databricks.yml`) after the
  app's first successful startup against `latlang_test`, then compare row
  counts against the historical source database before ever pointing it at
  the real `latlang`.
