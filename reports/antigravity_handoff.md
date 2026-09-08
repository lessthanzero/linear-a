# Antigravity Handoff — Linear A Evidence-First Reconciliation

**Prepared:** 2026-09-08  
**Working tree:** `/Users/sashakatin/Developer/linear-a`  
**Status:** implementation is complete for the agreed reconciliation scope: census engine, offline census explorer, and documentation. Do **not** discard the uncommitted worktree changes; this repository contains a larger, intentionally uncommitted body of work from the same effort.

## Read this first

Linear A remains undeciphered. Treat this repository as a research-inspection and hypothesis-management project. No module, metric, UI label, or generated report should be presented as a decipherment, translation, recovered language, verified grammar, or independently validated restoration system.

The active policy decisions are:

1. **E2 policy: abstain by default.** Distributional CV shape is descriptive context, not evidence for one missing syllabogram.
2. **Scope completed:** engine, explorer, and documentation.
3. **Provenance policy: snapshot-only disclosure.** Show document ID, token index, transliteration/glyph, source-manifest hash, and “primary-edition locator unavailable.” Do not invent GORILA page or object locators.
4. **Catalog hypotheses remain unverified.** A source audit may support a conventional form but does not validate the damaged-sign restoration, geographic route, or semantic interpretation.

## Current verified state

Run these commands from the repository root:

```bash
uv run pytest -q
uv run ruff check --select F,E9 src/linear_a/predictive/corpus_census.py src/linear_a/visualizer/workbench.py src/linear_a/cli/main.py tests/test_corpus_census.py tests/test_workbench.py
uv run linear-a census-lacunae --limit 10
uv run linear-a workbench --output reports/linear_a_workbench.html
```

At handoff time:

- `uv run pytest -q` → **66 passed**.
- Focused Ruff `F,E9` check → **passed**.
- `git diff --check` → **passed**.
- Generated workbench → `reports/linear_a_workbench.html`, about **2.5 MB**, because it embeds the complete 2,636-row census snapshot for offline filtering.

The current manifest-verified census is:

| Evidence status | Count | Meaning |
|---|---:|---|
| `ACCOUNTING_CONTEXT_E1` | 753 | Observed accounting context; no unique residual inferred. |
| `DETERMINISTIC_E3` | 0 | No independently verified residual calculation satisfies the census rule. |
| `SACRED_LITURGY_E4` | 19 | Curated recurring-form hypothesis. |
| `ATTESTED_TEMPLATE_E4` | 46 | Template/catalog hypothesis; unverified. |
| `OPEN_PHONOTACTIC_E2` | 545 | Context only; no sign proposal, completion, score, or confidence. |
| `IRRECOVERABLE_E0` | 1,273 | No proposal. |

Snapshot totals: **1,723 inscriptions**, **5,913 token groups**, **2,636 damaged-token entries**. Only E3/E4 entries are counted as candidate-supported: **65 / 2,636 (2.5%)**. That number is a census category total, not restoration accuracy.

The raw corpus is intentionally ignored by Git under `corpus/raw/`. It must match [`corpus/palaeography/corpus_source_manifest.yaml`](../corpus/palaeography/corpus_source_manifest.yaml) before the census command runs. The reviewed snapshot hashes are:

- `annotations.js`: `7ce1f87a98827d059a732cc00506c635b4d5f65b2d0e2f1592fc2b67827758cd`
- `LinearAInscriptions.js`: `4da8e1f9693d30880ee505e56541fc189add70605bad88436c44a8e11a57764c`

## What changed in this reconciliation

### 1. Corpus census engine

Primary file: [`src/linear_a/predictive/corpus_census.py`](../src/linear_a/predictive/corpus_census.py)

Before this change, the fallback E2 branch always emitted a fixed sign (`PA`, `TA`, or `TE`) plus a fixed Bayes factor and confidence. That behavior was removed.

The E2 branch now emits:

```python
recoverability_tier = "OPEN_PHONOTACTIC_E2"
suggested_infill = None
completed_word = None
bayes_factor = 1.0
epistemic_confidence = 0.0
```

The rationale states that distributional transitions do not uniquely identify a missing sign. E2 is excluded from `total_recoverable_count` and each site's candidate-supported total.

The YAML serializer now calls the status `open_phonotactic_e2` in metadata, includes `source_evidence_status` on entries, and records the primary-locator limitation. The current tracked export is [`corpus/palaeography/corpus_lacunae_census.yaml`](../corpus/palaeography/corpus_lacunae_census.yaml).

The engine also reads the toponym audit YAML directly, without importing `toponym_audit.py` (which would create a circular import through the lexical benchmark). Its current statuses are:

- `PA-I-TO` / `HT120`: `attested` conventional-form record.
- Other catalog toponymic forms: `unaudited_catalog_hypothesis`.
- Other catalog rows: `unverified_catalog_hypothesis`.

Do not promote these strings to verification of a restoration.

### 2. CLI census output

Primary file: [`src/linear_a/cli/main.py`](../src/linear_a/cli/main.py)

`linear-a census-lacunae` now labels E2 as “Open phonotactic contexts,” says it has no sign proposal or score, and explicitly says candidate-supported totals exclude open E2 contexts.

Useful commands:

```bash
uv run linear-a census-lacunae --tier E2 --limit 25
uv run linear-a census-lacunae --site "Haghia Triada" --limit 25
uv run linear-a census-lacunae --export corpus/palaeography/corpus_lacunae_census.yaml
```

The `--tier` filter is substring-based, so `--tier E2` finds `OPEN_PHONOTACTIC_E2`.

### 3. Offline workbench explorer

Primary source: [`src/linear_a/visualizer/workbench.py`](../src/linear_a/visualizer/workbench.py)  
Generated artifact: [`reports/linear_a_workbench.html`](linear_a_workbench.html)

A new **10. Corpus Census** tab loads the tracked YAML snapshot. It does not run the census or read `corpus/raw/` during workbench generation. It supports:

- search by document ID, transliteration token, or glyph;
- filters for site, carrier, and evidence status;
- all 2,636 snapshot rows offline;
- source-manifest hash prefixes; and
- explicit “Primary-edition locator unavailable in this local source snapshot.”

`OPEN_PHONOTACTIC_E2` rows display **“Abstained: no sign proposed.”** Source-evidence labels appear for catalog-derived rows when applicable. The output uses an HTML escaping helper for rendered snapshot fields.

Do not move the census data source from the tracked YAML to a live raw-source invocation: portability without ignored inputs is an explicit requirement.

### 4. Documentation reset

The prior monograph and walkthrough contained historical project claims that exceeded the current evidence state. They were replaced with concise current documents:

- [`reports/linear_a_monograph.md`](linear_a_monograph.md)
- [`walkthrough.md`](../walkthrough.md)

Both explain the E2 abstention policy, snapshot provenance, toponym audit limits, null/pending benchmark status, and reproduction commands. Keep this careful framing if editing them.

## Other completed work in the current uncommitted change set

The reconciliation sits on top of earlier work that is also present but not committed. Preserve it while extending the project:

| Area | Main implementation | Current result / boundary |
|---|---|---|
| Structural parser | `src/linear_a/grammar/pcfg_engine.py`, `tree_renderer.py`; `linear-a syntax <id>` | Rule-based labels for curated transcription only; no grammar or meaning claim. |
| Source-linked restoration benchmark | `src/linear_a/predictive/source_linked_benchmark.py`, `restoration_evaluation.py`; `linear-a evaluate-restorations` | 23 records are unadjudicated; no independent scored accuracy. |
| Lexical-pair benchmark | `src/linear_a/skeptic/substratum_filter.py`; `linear-a substratum` | 0/12 qualifying published Linear A–Linear B lexical pairs; no empirical score. |
| Cretan toponym audit | `src/linear_a/predictive/toponym_audit.py`; `linear-a toponym-audit` | One attested `PA-I-TO` conventional correspondence, one disputed `DA-WO` interpretation; no restoration/route inference. |
| Catalog/solver caveats | `corpus/palaeography/lacunae_catalog.yaml`, `multilateral_solver.py` | Toponymic catalog entries are source-linked but unadjudicated or unaudited hypotheses, not exact matches. |

Relevant artifacts and inputs:

- [`corpus/palaeography/source_linked_benchmark.yaml`](../corpus/palaeography/source_linked_benchmark.yaml)
- [`corpus/palaeography/restoration_evaluation.yaml`](../corpus/palaeography/restoration_evaluation.yaml)
- [`corpus/palaeography/cretan_toponym_audit.yaml`](../corpus/palaeography/cretan_toponym_audit.yaml)
- [`corpus/lexicons/linear_b_correspondence_benchmark.yaml`](../corpus/lexicons/linear_b_correspondence_benchmark.yaml)
- [`reports/source_linked_review_handoff.json`](source_linked_review_handoff.json)

## Known constraints and non-goals

- The raw source export has no per-token primary-edition locators. Never manufacture citations based on embedded image references or document IDs.
- The source-linked restoration records need qualified independent adjudication before precision/recall or “exact match” claims are defensible.
- `PA-I-TO` published conventional-form attestation is narrower than a verification of the project’s damaged token completion.
- The workbench contains older visual sections that describe project-curated material strongly. The new documentation and census tab are evidence-first; if revising older sections, reduce claims rather than adding new inferred results.
- Full Ruff currently reports pre-existing/style modernization findings across inherited code. The focused `F,E9` check is the completed regression check. Do not mass-reformat unrelated modules unless the task explicitly includes that cleanup.
- No commit or remote push has been made. The user has not asked for one.

## Recommended next work, in priority order

1. **External review packet, not new infills.** Expand `reports/source_linked_review_handoff.json` into a reviewer-facing packet with image/object references supplied by a qualified source holder. Capture accept/reject/uncertain adjudications as a separate dataset; do not overwrite the existing hypotheses.
2. **Provenance schema.** Add optional, source-authenticated primary-edition locator fields to the census schema. Populate only from a reviewed bibliography or licensed edition. Keep `primary_locator_available: false` for the current snapshot.
3. **Workbench claim audit.** Review the older workbench tabs and soften or remove labels that say “proof,” “recovered,” “exact restoration,” “autonomous,” or semantic/function assertions that exceed the underlying curated data. Preserve purely mechanical arithmetic checks where source data supports them.
4. **Catalog reclassification.** Split catalog-derived E4 values into more precise statuses such as `UNVERIFIED_CATALOG_TEMPLATE_E4` versus source-linked-but-unadjudicated records. This is a schema/UI change; do it only with migration tests and a regenerated snapshot.
5. **Independent evaluation pipeline.** Once external adjudication exists, add frozen train/test splits, reviewer blinding, disagreement handling, and explicit abstention metrics. Do not use catalog agreement as evaluation.
6. **Snapshot maintenance.** If local raw inputs change, review the source, update `corpus_source_manifest.yaml`, regenerate `corpus_lacunae_census.yaml`, regenerate the workbench, and run the test suite together.

## Suggested first Antigravity prompt

> Read `reports/antigravity_handoff.md` and inspect the uncommitted worktree without resetting it. Verify `uv run pytest -q`, then audit the older workbench prose for claims stronger than the evidence-first policy. Produce a small, test-backed patch that only softens unsupported UI wording; do not alter census classifications, raw-source provenance, or generated data until the wording review is complete.
