# Git LFS audit report (2026-10-01)

Read-only audit: nothing was modified or deleted in any repository.

## Summary

| Repository | Visibility | LFS in HEAD | Verdict |
|---|---|---|---|
| `JulesGourio/parsing-ocr` | public | **about 6.45 GB** | Main consumer |
| `JulesGourio/Parsing` | public | **about 4.17 GB** | Second consumer |
| `JulesGourio/parsing-ocr2` | public | none (empty repository) | Nothing to do |
| `JulesGourio/test_ocr` | public | none | Nothing to do |
| `JulesGourio/uploadgit` | public | none | Nothing to do |
| `JulesGourio/side_project` | public | none | Nothing to do |
| `running-coach`, `IDMProject`, `Dan-Project`, `AIF`, `Time-LLM`, `ProjetKaggleModIA` | public | no LFS rule | Nothing to do (AIF contains two 45 MB MNIST files, stored as regular git objects) |
| 12 private repositories and `UtalTIPE/AIFproject` | private / org | **not audited** | Needs push access (see below) |

Combined, the two repositories hold about 10.6 GB of LFS objects in HEAD alone, which exceeds the free quota (10 GiB of storage).
Older commits may add more: LFS counts every version ever pushed, and this audit only sees the current tree.

## Details

### `parsing-ocr` (about 6.45 GB)
- `model_part_aa`, `_ab`, `_ac`: 1.8 GB each, `model_part_ad`: 963 MB. A split model, about 6.3 GB, the bulk of the usage.
- `assets/long-horizon-ocr.gif` (78 MB), `wheel/sglang-...whl` (12 MB), `assets/baidu.png`.
- Tracked patterns: `*.safetensors`, `*.mov`, `*.mp4`, `*.whl`, `*.png`, `*.gif`, `*.pdf`, `*.bin`, `*.pt`, `*.msgpack`, `model_part_*`.
- Looks like a copy of a public OCR model repository (assets such as `Unlimited-OCR.pdf`, a `tokenizer.json.bak`).

### `Parsing` (about 4.17 GB)
- `docling/paquets_python_offline/*.whl`: offline Python packages (torch 506 MB, nvidia cublas 404 MB, cudnn 349 MB, cufft, cusolver, nccl, triton, ...), about 2.7 GB in total.
- `docling/docling_models/**`: Docling model weights (`model.safetensors` 602 MB, tableformer, layout heron, ...), about 1.5 GB in total.
- All of this is reproducible: wheels come from PyPI and models from Hugging Face.

## Recommendations (nothing applied yet)

1. **Do not keep these files in git.** Wheels can be re-downloaded (`pip download`) and models fetched from Hugging Face. For a Databricks offline setup, store them in a Unity Catalog volume instead.
2. **Freeing LFS quota is not done with a commit.** Removing the files in a new commit or running `git lfs prune` does not release the server-side storage. GitHub only frees it when:
   - the repository is deleted (then recreated without the LFS files), or
   - GitHub Support purges the orphaned LFS objects.
3. **Option A (simplest):** delete `parsing-ocr` (a copy of an external project, easy to re-clone) and recreate `Parsing` without `docling/paquets_python_offline` and `docling/docling_models`. This frees about 10.6 GB.
4. **Option B (keep history):** remove the files from the tree and from history (`git filter-repo`), then ask GitHub Support to purge the LFS objects.
5. Add a `.gitignore` entry for `*.whl`, `*.safetensors`, `model_part_*` to prevent a recurrence.

## Not audited

The 12 private repositories (`RAG-Chunking`, `Dashboard_TradeRepublic`, `Obsidian-AI`, `Molecular_Energy_Prediction`, `MovieRecommenderAIF`, `DataAssimilation`, `TimeSeriesAnalysis`, `Defi_IA_Carrefour`, `ProjetBigdata`, `Partage-Airbus`, `GestionDeProjet4A`) and `UtalTIPE/AIFproject` require a push-level attachment. Their LFS usage can also be read directly in GitHub: Settings > Billing and plans > Git LFS Data.
