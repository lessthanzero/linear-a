# Linear A Evidence-First Computational Workbench

**Project status:** research infrastructure and bounded structural analysis. Linear A remains undeciphered; none of the outputs in this repository supplies a translation or a recovered language.

**Snapshot:** September 2026. The reproducible corpus census is a tracked export from a manifest-verified local snapshot. Its raw inputs are intentionally ignored by Git; the exported data records the corresponding file hashes but has no primary-edition page or object locators.

## What the project currently supports

The workbench exposes a set of constrained tools for inspecting curated data and testing narrow computational claims:

- exact arithmetic checks on the project’s curated ledger examples;
- a rule-based structural pattern display for the curated interlinear documents;
- source-linked restoration and toponym audit records, kept separate from exploratory catalog agreement;
- a strict published Linear A–Linear B lexical-pair benchmark; and
- a corpus-wide damaged-token census from a manifest-verified local snapshot.

These are aids to audit and hypothesis management. They are not decipherment methods and do not establish grammar, phonetic values, lexical meanings, or historical routes.

## Census protocol and result

`linear-a census-lacunae` scans the local snapshot only when both raw files match [`corpus_source_manifest.yaml`](../corpus/palaeography/corpus_source_manifest.yaml). The committed [`corpus_lacunae_census.yaml`](../corpus/palaeography/corpus_lacunae_census.yaml) is the portable snapshot used by the offline workbench.

The current snapshot contains 1,723 inscriptions, 5,913 token groups, and 2,636 damaged-token entries. Its categories are evidence statuses rather than claims of restored text:

| Status | Count | Treatment |
|---|---:|---|
| `ACCOUNTING_CONTEXT_E1` | 753 | Observed accounting context; no missing value inferred. |
| `DETERMINISTIC_E3` | 0 | No independent residual calculation met the census rule. |
| `SACRED_LITURGY_E4` | 19 | Curated recurring-form hypothesis. |
| `ATTESTED_TEMPLATE_E4` | 46 | Template or catalog hypothesis; unverified. |
| `OPEN_PHONOTACTIC_E2` | 545 | Descriptive context only; no sign, completion, or score is emitted. |
| `IRRECOVERABLE_E0` | 1,273 | No proposal. |

The 65 E3/E4 entries (2.5% of damaged tokens) are the only rows counted as candidate-supported by the current census rules. This is a bookkeeping measure, not a restoration accuracy result. E2 was previously populated by a fixed sign heuristic; it is now an explicit abstention because broad CV distributional shape cannot identify a missing syllabogram.

## Provenance and interpretation limits

Each census row retains document identifier, token index, transliteration, glyph, site, carrier, and the evidence rationale. The snapshot provenance includes hashes for the two local source files. It deliberately states **“primary-edition locator unavailable”** rather than fabricating GORILA or other edition references.

Curated catalog entries remain unverified hypotheses. For catalog entries marked toponymic, the census additionally exposes a source-evidence status: `PA-I-TO` has a published conventional-form correspondence in the audit; the other catalog toponymic forms are unaudited catalog hypotheses. The audit does not verify a damaged sign restoration or establish a route, administrative function, or translation.

The toponym audit records only one attested correspondence (`PA-I-TO` / Linear B `PA-I-TO` / Greek *Phaistos*) and one disputed geographic interpretation (`DA-WO`). It uses published source records, including [Uchitel (2016)](https://gredos.usal.es/handle/10366/133947) and [Monti (2024)](https://doi.org/10.1515/kadmos-2024-0001), but does not turn those records into a general Linear A–Linear B decipherment claim.

The published lexical-pair benchmark currently finds zero qualifying pairs under its stated criteria. The source-linked restoration benchmark contains unadjudicated records and therefore reports no scored accuracy. Those null or pending outputs are intentional safeguards against treating project-curated agreement as external validation.

## Reproducible commands

```bash
uv run linear-a census-lacunae --export corpus/palaeography/corpus_lacunae_census.yaml
uv run linear-a toponym-audit --export reports/cretan_toponym_audit.json
uv run linear-a substratum --export reports/linear_b_correspondence_benchmark.json
uv run linear-a evaluate-restorations --export reports/restoration_evaluation.json
uv run linear-a workbench --output reports/linear_a_workbench.html
uv run pytest -q
```

The offline [`linear_a_workbench.html`](linear_a_workbench.html) includes a searchable census explorer with site, carrier, and evidence-status filters. It loads the tracked YAML snapshot and therefore remains inspectable without access to the ignored local raw corpus.

## Review priorities

1. Replace snapshot-level provenance with source-specific scholarly locators where rights and source access permit.
2. Obtain independent adjudication before reporting restoration precision or recall.
3. Keep catalog-derived suggestions separate from source-backed evidence and from any semantic interpretation.
4. Treat all structural labels as descriptions of the curated transcription, not grammatical analyses.
