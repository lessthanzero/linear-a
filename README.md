# Linear A Computational Laboratory

**Status (v0.2.0 research preview):** Linear A remains undeciphered. This repo is an evidence-bounded inspection harness — not a translation, recovered language, or confirmed cross-script homology. See [SCIENTIFIC_LIMITATIONS.md](SCIENTIFIC_LIMITATIONS.md) and [NOTICE](NOTICE).

Computational research laboratory and epigraphic analysis harness for the undeciphered Bronze Age Minoan Linear A script.

Enforces the **Linear A Decipherment Protocol (LADP)** with multi-node distributed compute across Apple Silicon macOS and Fedora Linux over Tailscale, and a multi-model consensus jury (local Ollama models + cloud Gemini).

## Cautious research tools

`linear-a syntax HT_009` and `linear-a syntax IO_Za_002` render rule-based structural-pattern hypotheses for the repository's curated readings. Pattern labels and coverage describe observed transcriptional structure only; they do not establish grammar, language, meaning, or decipherment.

`linear-a census-lacunae` catalogs damaged tokens in an optional local SigLA/GORILA-derived snapshot. It requires `corpus/raw/annotations.js` and `corpus/raw/LinearAInscriptions.js` to match [corpus_source_manifest.yaml](corpus/palaeography/corpus_source_manifest.yaml). Those source files are intentionally untracked. Update the manifest and generated census snapshot together after reviewing a new local source export.

`linear-a evaluate-restorations` scores only `accepted` records from the [source-linked benchmark](corpus/palaeography/source_linked_benchmark.yaml). The current citation-only bootstrap has 23 `unadjudicated` records and therefore reports zero scoreable references. Use `--review-handoff <path>` to export source-linked reviewer records, and `--include-exploratory` to display the separate project-curated exploratory report.

### Published Linear A–Linear B lexical-pair benchmark

`linear-a substratum` evaluates only pre-registered, published lexical Linear A–Linear B comparisons. The first evidence review found no qualifying lexical pair after excluding names and toponyms and requiring an explicit published comparison, a source locator, and a Linear B tablet attestation. It therefore reports a documented shortfall and does not calculate a score. This is not evidence against any language relationship; it prevents unsupported comparisons from being scored as data.

The tracked benchmark at `corpus/lexicons/linear_b_correspondence_benchmark.yaml` cites Rafaela Freire de Abreu e Souza (2022), *Aspects of Non-Greek Vocabulary in Mycenaean Greek*, pp. 13–15 and 57–59, and uses DĀMOS for Linear B corpus and attestation context.

### Source-linked Cretan toponym audit

`linear-a toponym-audit` reports published toponym evidence separately from catalog hypotheses. The initial audit records `PA-I-TO` as a cross-script conventional-form correspondence and records the `DA-WO`/Hagia Triada attribution as disputed. It does not validate damaged-sign restorations, routes, morphology, translation, or the language of Linear A. The catalog now labels all other toponymic forms as unaudited hypotheses.
