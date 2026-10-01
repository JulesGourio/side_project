"""Translation engine — the document-translation pipeline modules owned by
the app (extract, audit, langdetect, glossary_io, fit_check, length_adapt,
rebuild, validate).

Originally developed and validated as a CLI prototype under Translator/tools/
(removed 2026-07-20 once confirmed unused — the web tab fully replaced the
interactive CLI workflow); these copies are the ones the server runs.
Differences vs the prototype were import mechanics only: package-relative
imports instead of sys.path insertion. The CLI `main()` functions and
`__main__` blocks are inert here (their job_utils imports are lazy and never
triggered by the app).

server/services/processors/translation.py is the in-memory adapter the
translate router calls; it composes these modules against uploaded bytes
instead of CLI file paths.
"""
