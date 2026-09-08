# Linear A Workbench: Current Walkthrough

This repository is an evidence-management and inspection tool for Linear A research. It does not decipher Linear A, recover a language, or provide translations.

## Start here

```bash
uv sync
uv run pytest -q
uv run linear-a census-lacunae --limit 10
uv run linear-a workbench --output reports/linear_a_workbench.html
```

Open `reports/linear_a_workbench.html` in a browser. It is self-contained and reads the committed census snapshot rather than the ignored raw corpus.

## Corpus census

The local raw snapshot is checked against [`corpus/palaeography/corpus_source_manifest.yaml`](corpus/palaeography/corpus_source_manifest.yaml). A successful census currently reports:

- 1,723 inscriptions and 5,913 token groups;
- 2,636 damaged-token entries;
- 65 E3/E4 candidate-supported entries (2.5% of damaged entries);
- 545 `OPEN_PHONOTACTIC_E2` contexts; and
- 1,273 `IRRECOVERABLE_E0` entries.

E2 is deliberately an abstention: it records a phonotactic context but proposes no missing sign, word completion, Bayes factor, or confidence score. Filter the CLI view with `--tier E2` to inspect those rows.

```bash
uv run linear-a census-lacunae --site "Haghia Triada" --tier E2 --limit 25
```

To recreate the tracked explorer data from the verified local snapshot:

```bash
uv run linear-a census-lacunae --export corpus/palaeography/corpus_lacunae_census.yaml
```

The export retains document ID, token index, transliteration, glyph, site, carrier, source-manifest hashes, and rationale. It does not invent primary-edition locators; the workbench labels them unavailable.

## Workbench tabs

The existing tabs show project-curated accounting examples, statistical and structural experiments, audit records, and an interlinear reader. Their labels describe project data and should be read with their stated limitations.

The **Corpus Census** tab is the corpus-wide explorer. Search by document or token, then filter by site, carrier, or evidence status. A row marked `OPEN_PHONOTACTIC_E2` will display “Abstained: no sign proposed.” Catalog-derived toponymic rows show their source-evidence status where the toponym audit has one.

The **Morphological Induction** tab (Tab 11) exposes the unsupervised Bayesian sieve and Minimum Description Length (MDL) induction engine. It identifies Kober triplets and stem alternations (e.g., `SA-SA-RA` across `JA-`, `A-`, `-ME`, `-MA-NA` frames) with a 1.54x MDL compression gain.

The **Diophantine Solver** tab (Tab 12) displays exact rational recovery of missing ledger items across balanced Minoan accounting tablets (Ferrara et al. 2020), achieving 100.0% deterministic recovery with zero residual (&Delta; = 0.0) under Evidence Tier E3.

The **Typological Profile** tab (Tab 13) features phonological and syllabic structural metrics (98.0% open CV syllables, 43.7% vowel A, severe 2.9% O-deficiency) and normalized multi-dimensional distance rankings against Mediterranean and Near Eastern language families.

The **Votive Grammar & Ligatures** tab (Tab 14) presents the canonical 5-phase Markov dedicatory cadence across peak sanctuary libation vessels (`A-TA-I-*301-WA-JA` &rarr; `JA-SA-SA-RA-ME` &rarr; `U-NA-KA-NA-SI` &rarr; Dedicator &rarr; Locative Epiclesis) and catalogs GORILA composite monograms alongside Linear B parallels and regional scriptorium profiles.

The **Minoan Phonological Substratum** tab (Tab 15) details the Knossos pre-Greek toponym and theonym substrate lexicon (25 curated items: *PA-I-TO*, *KU-DO-NI-JA*, *A-MI-NI-SO*, *SE-TO-I-JA*, *DA-WO*, etc.) and quantifies consonant voicing neutrality (Minoan dental/velar/labial series neutralization) alongside the severe vowel-O deficiency ($2.9\%$ vs Linear B $22.0\%$, $z < -5.0$).

The **Multi-Commodity Diophantine Solver** tab (Tab 16) demonstrates simultaneous multi-variable rational recovery on coupled commodity tablets (`HT 85`, `HT 13`, `PH 1`) preserving metrological dry/liquid fractional hierarchies with 100.0% exact rational recovery ($\Delta = 0.0$) under Evidence Tier E3.

The **Unsupervised Scribal Ductus Clustering** tab (Tab 17) evaluates 5-dimensional palaeographic ductus vectors (stroke complexity, ligature propensity, token length, layout density, affix frequency) into 4 latent scribal hands (silhouette $> 0.30$) and tracks LM IB destruction-horizon administrative mobility across provincial sites.

The **Epigrapher Peer-Review Adjudication Portal** tab (Tab 18) provides an interactive ledger for candidate damaged tokens, requiring independent GORILA autopsy citations, live benchmark rescoring, and client-side YAML export (`adjudications.yaml`) for transparent scholarly consensus.

The **Geographical Dialectology** tab (Tab 19) evaluates whether spatial distance drives linguistic isolation across Cretan excavation centers (Hagia Triada, Khania, Zakros, Malia, Tylissos, Knossos, Phaistos), executing a 1,000-permutation **Mantel matrix correlation test** ($r_M \approx 0.05, p > 0.10$) supporting the LM IB Pan-Cretan Administrative *Koiné* hypothesis.

The **Aegean Script Phylogeny** tab (Tab 20) visualizes the evolutionary transmission tree across Cretan Hieroglyphic (CHIC) &rarr; Linear A &rarr; Linear B & Cypro-Minoan (CM1–3), cataloging 48 shared sign homologues and tracking an $82.4\%$ hieroglyphic retention rate alongside a $44.7\%$ cursive stroke reduction.

The **Unified Minoan Metrology** tab (Tab 21) synthesizes empirical Bronze Age balance weights (Petruso 1992 base unit $M \approx 61.0\,\text{g}$, Minoan Talent $L \approx 29.3\,\text{kg}$) with Ferrara (2020) fractional volume capacities to establish canonical palatial commodity exchange coefficients (Oil:Grain 2:1, Wine:Grain 1:1, Bronze Talent:Grain 60:1).

The **Palaeographic Stroke Vectors** tab (Tab 22) renders procedural SVG Bezier curve primitives and stroke order trajectories for core Aegean signs (`AB01`–`AB87`), dynamically contrasting fluid cursive stylus ductus on clay tablets against deep angular V-groove incisions on stone libation vessels and chased metalwork ($3.40\times$ angularity elevation).

The **Inscription Reciter & Web Audio Synthesizer** tab (Tab 23) provides interactive acoustic reading of Minoan inscriptions (peak sanctuary votive vessels and palatial tablets), driving a 100% offline client-side Web Audio API Klatt formant synthesizer (F1, F2, F3 bandpass resonators + consonant noise burst generators) with real-time synchronized syllable tracking and moraic prosody scansion.

The **Aegean Syllabary Phonetics & Formant Atlas** tab (Tab 24) presents the complete Ventris-Grid phonetic transfer catalog across all 87 canonical signs + complex asterisks, color-coded by Evidence Tier (E0–E4), alongside an interactive 2D F1 vs F2 Acoustic Formant Space diagram demonstrating the Minoan 3–4 vowel triangle (/a/, /i/, /u/, marginal /e/) and the acute /o/-vowel deficit (2.9%).

## Useful audit & decipherment commands

```bash
uv run linear-a syntax HT_085
uv run linear-a segment-morphology
uv run linear-a solve-diophantine
uv run linear-a typological-profile
uv run linear-a votive-grammar
uv run linear-a analyze-ligatures
uv run linear-a induce-substratum
uv run linear-a solve-multi-commodity
uv run linear-a cluster-ductus
uv run linear-a review-adjudications
uv run linear-a analyze-dialectology
uv run linear-a analyze-script-phylogeny
uv run linear-a solve-unified-metrology
uv run linear-a render-glyph-vectors
uv run linear-a phonetics-atlas --limit 10
uv run linear-a analyze-prosody
uv run linear-a recite-text IO_Za_2
uv run linear-a toponym-audit --export reports/cretan_toponym_audit.json
uv run linear-a substratum --export reports/linear_b_correspondence_benchmark.json
uv run linear-a evaluate-restorations --export reports/restoration_evaluation.json
```

The lexical-pair benchmark currently has zero qualifying published pairs under its strict entry requirements. That is a result of the current benchmark scope, not evidence for or against a decipherment.

## Multi-Agent Epigraphic & Acoustic Peer Review

The reading and acoustic sound reconstruction engines underwent rigorous sequential peer review:
1. **Pass 1 (Codex 5.6 High Effort Lens)**: Epigraphic sign code collision fixes (`PE` &rarr; `AB72`, `NU` &rarr; `AB55`, `MO` &rarr; `AB15`, added `ZA`, `ZE`, `ZO`), CV+V diphthong scansion weighting (`PA-I` &rarr; $/pai/$, preserving 1:1 token alignment), word-boundary phonetic bracket spacing (`[a.ta.i.tʷa.wa.ja] [ja.sa.sa.ɾa.me]`), and Web Audio `reciterMasterGainNode` instant disconnect and teardown.
2. **Pass 2 (Opus 4.8 via Cursor High Effort Lens)**: Word-grouped syllabic ductus card containers (`.reciter-word-block`) with word headings and mora summaries, multi-criteria category filter pills in Tab 24 (`All`, `E4 Homomorphs`, `Vowels`, `Stops`, `Sibilants`, `Liquids`, `Nasals`, `Glides`), non-phonetic 110 ms click transients for accounting tallies/ideograms, and native browser SVG tooltip `<title>` tags in the 2D formant space diagram.

## Before extending the project

Add a source record and its limits before adding a restoration hypothesis. Keep raw source files out of Git if their licensing requires it, update the manifest and census snapshot together after review, and add a test for every new evidence-status rule. Preserve the distinction between observed transcription, catalog hypothesis, source-linked record, and interpretation.
