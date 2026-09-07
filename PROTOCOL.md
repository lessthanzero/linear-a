# Linear A Decipherment Protocol (LADP)

**Version**: 1.0 (Laboratory Edition with Epistemic & Mathematical Enhancements)  
**Derived From**: ChatGPT Brief v1.0, GORILA (Godart & Olivier 1976–1985), SigLA, and Phaistos Disc Computational Laboratory Harnesses.

---

## 0. Objective

Infer the linguistic system underlying Linear A without assuming its language family or equating Linear A with Linear B.

**Primary Target Pipeline**:
$$\text{GLYPH} \longrightarrow \text{PHONETIC VALUE} \longrightarrow \text{MORPHOLOGY} \longrightarrow \text{SYNTAX / CONTEXT} \longrightarrow \text{SEMANTICS}$$

**Core Mandate**: Do NOT optimize initially for translation. Translation comes last. Optimize for:
1. Corpus-wide explanatory power
2. Predictive power on held-out inscriptions
3. Minimum assumptions (Minimum Description Length)
4. Reproducibility
5. Falsifiability under Shannon unicity and Monte Carlo null controls

---

## 1. Core Principle & Layer Separation

Evidence layers are strictly non-fungible:
$$\text{GLYPH} \neq \text{SIGN-ID} \neq \text{PHONETIC-VALUE} \neq \text{WORD} \neq \text{MORPHEME} \neq \text{MEANING}$$

* **Rule of Non-Promotability**: Never collapse uncertain layers. An inference at a higher layer can never be treated as an immutable observation at a lower layer.
* **Linear B Role**: Linear B correspondences provide candidate phonetic priors ($L_2$), NOT guaranteed Linear A readings and NEVER Greek translations.
* **Phaistos Disk Firewall**: The Phaistos Disk dataset (`CORPUS_PHAISTOS_DISK`) is quarantined from Linear A (`CORPUS_LINEAR_A`). Cross-script correspondence must be independently established through statistical and formal paleographic proofs.

---

## 2. Corpus Data Model

```
INSCRIPTION
  id: unique_id (e.g. HT_009, PH_001, IO_Za_002)
  site: Phaistos | Hagia_Triada | Knossos | Malia | Zakros | Khania | other
  date_range: MM II | MM III | LM IA | LM IB
  object_type: tablet | roundel | nodule | libation_vessel | metal_axe | dipinto
  source: GORILA | SigLA | Younger
  text_units: [TEXT_UNIT]
  confidence: float (0.0 .. 1.0)

TEXT_UNIT
  id: unique_id
  inscription_id: str
  line_index: int
  sequence: [SIGN]
  word_boundaries: [int]
  damage_mask: [bool]
  numerals: [int]
  fractions: [str] (e.g. J, E, F, K, A, H)
  logograms: [str] (e.g. GRA, VIN, FIC, CYP, VIR)
  divider_positions: [int]

SIGN
  glyph_id: canonical_glyph_id (e.g. AB01, AB04, A301)
  variant_ids: [str]
  sign_class: syllabic | logogram | numeral | fraction | punctuation | unknown
  linearB_correspondence:
    candidate: str (e.g. "da", "te")
    confidence: float (0.0 .. 1.0)
  phonetic_candidates:
    - value: str
      confidence: float (0.0 .. 1.0)

TOKEN
  token_id: str
  inscription_id: str
  glyph_sequence: [str]
  phonetic_candidates: [str]
  position: initial | medial | final | standalone
  preceding_tokens: [str]
  following_tokens: [str]
  numerical_context: Optional[dict]
  semantic_context: administrative | religious | commodity | person | place | unknown

HYPOTHESIS
  id: str (e.g. H-001)
  type: phonetic | morphological | syntactic | semantic | language_family | cross_script
  claim: str
  evidence_ids: [str]
  predictions: [str]
  counterexamples: [str]
  assumptions: [str]
  exceptions: [str]
  confidence: float (0.0 .. 1.0)
  status: active | weak | falsified | supported
```

---

## 3. Evidence Hierarchy

Evidence is ranked in ascending order of epistemic weight:
* **E0**: Visual / glyph observation (incised clay marks, stylus burrs, strokes).
* **E1**: Repeated corpus pattern (unigram/bigram token recurrence).
* **E2**: Positional / distributional constraint (initial/medial/final clustering).
* **E3**: Numerical / accounting constraint (commodity tallies, fractional sums, ledger totals).
* **E4**: Cross-inscription recurrence (multi-site lexical stability).
* **E5**: Morphological / systemic evidence (root-and-affix paradigms, inflectional alternations).
* **E6**: Semantic / contextual convergence (consistent administrative or votive environments).
* **E7**: Independent predictive confirmation (prediction holds on held-out inscriptions).

**Conflict Resolution Rule**:
$$\mathbf{E_3 \text{ to } E_7} \gg \mathbf{E_0 \text{ to } E_2}$$
When layers conflict, mathematical, morphological, and distributional constraints strictly override phonetic or visual impressions.

---

## 4. Phonetic Rules & Syllabic Loss Modeling

1. Maintain all plausible phonetic readings with explicit Bayesian prior confidences.
2. Use Linear B correspondences as prior evidence only ($L_2$), not ground truth.
3. Model Aegean syllabic information loss explicitly:
   * Consonant coda omission (word-final /s/, /n/, /r/ unwritten in open CV syllabaries).
   * Vowel neutralizations (/e/ vs /i/, /o/ vs /u/).
   * Voiced / voiceless / aspirated stop collapse (e.g. /k/, /g/, /kh/ written with identical $K$-series signs).
   * Consonant cluster splitting via dummy vowels (e.g. *kra* written *ka-ra*).
4. **The Epistemic Ban**: Never introduce or adjust a phonetic value solely because it produces a desired lexical word in a candidate language.

---

## 5. Morphology Discovery

Corpus-first search without presupposing Indo-European, Semitic, or Anatolian grammar:
* Detect recurring prefixes (`ja-`, `a-`, `u-`, `da-`).
* Detect recurring suffixes (`-te`, `-ne`, `-re`, `-se`, `-me`, `-ti`, `-na`).
* Isolate stems and inflectional alternations (e.g. $X \to X\text{-A} \to X\text{-TE} \to X\text{-NE}$).
* Candidate morphology must satisfy:
  $$\text{Recurrence} + \text{Distributional Consistency} + \text{Compositional Invariance}$$

---

## 6. Semantic Inference

Contextual constraints must precede lexical translation:
* For token $X$: $\text{Context}(X) \longrightarrow \text{Candidate Semantic Class}$.
* A token preceding a numeral or commodity ideogram is classified structurally as an **accounting descriptor**, **recipient**, or **provenance**, NOT assigned a dictionary translation.

---

## 7. Numerical & Fractional Constraint Engine

Accounting texts represent high-fidelity mathematical ground truth:
* Implement the modern mathematical fractional consensus (Ferrara et al. 2020):
  $$J = \frac{1}{2}, \quad E = \frac{1}{4}, \quad F = \frac{1}{8}, \quad K = \frac{1}{16}, \quad A = \frac{1}{6} \text{ (or } \frac{1}{3}\text{)}, \quad H = \frac{1}{12}$$
* Test ledger balance:
  $$\sum \text{Inputs} \equiv \text{KU-RO (Total)}$$
* Infer commodity classes, fractional units, and transaction roles from mathematical consistency.

---

## 8. Distributional Analysis & Information Theory

For every token and sign sequence:
* Compute unigram entropy $H(X)$, conditional entropy $H(Y|X)$, and topological entropy $H_{top}$.
* Calculate Pointwise Mutual Information (PMI), n-gram transition probabilities, and cluster affinities.
* Calculate the **Shannon Unicity Distance ($U$)**:
  $$U = \frac{H(K)}{D_L}$$
  Any claim where model degrees of freedom exceed corpus information capacity is rejected as an **overfit combinatorial illusion**.

---

## 9. Regional Stratification Test

Partition the corpus by geographical provenances:
* **Central South**: Phaistos (PH), Hagia Triada (HT)
* **North / Central**: Knossos (KN), Malia (MA), Tylissos (TY)
* **West**: Khania (KH)
* **East**: Kato Zakros (ZA), Palaikastro (PK)

Test for shared administrative core vs regional dialectal or scribal divergences. Do NOT presuppose a single monolithic language across all periods (MM II to LM IB).

---

## 10. Language-Family Hypotheses (Late-Stage Protocol)

Language identification is evaluated only after structural, morphological, and numerical patterns are established.
* No language family receives privileged status (Anatolian/Luwian, Semitic, Tyrsenian, Hurro-Urartian, Archaic Greek, Minoan Isolate).
* A proposed cognate is valid if and only if it satisfies:
  1. Plausible phonological regularities;
  2. Plausible morphology;
  3. Plausible semantic convergence;
  4. Cross-inscription recurrence;
  5. Statistical survival against randomized synthetic null dictionaries (low False Positive Rate).

---

## 11. Quantitative Hypothesis Scoring Equation

Every hypothesis receives a deterministic score $S$:
$$S = + 3.0 \cdot \text{structural\_fit} + 3.0 \cdot \text{corpus\_coverage} + 3.0 \cdot \text{predictive\_success} + 2.0 \cdot \text{numerical\_fit} + 2.0 \cdot \text{morphological\_fit} + 1.5 \cdot \text{contextual\_fit} + 1.0 \cdot \text{regional\_fit} + 1.0 \cdot \text{recurrence} - 3.0 \cdot \text{arbitrary\_assumptions} - 3.0 \cdot \text{exceptions} - 2.0 \cdot \text{phonetic\_deviations} - 2.0 \cdot \text{unsupported\_semantic\_leaps} - 4.0 \cdot \text{ad\_hoc\_rules}$$

Normalized to $0 \dots 100$:
* **0–20**: Speculative / Falsified
* **20–40**: Weak
* **40–60**: Plausible
* **60–75**: Strong
* **75–90**: Very Strong
* **90–100**: Exceptional (requires independent blind replication)

---

## 12. Minimum Description Length (MDL) Rule

$$\text{Best Hypothesis} = \arg\max \left[ \text{Explanatory Compression} - \text{Model Complexity} \right]$$
Penalize special-case readings, arbitrary sign values, unexplained exceptions, and dialect-switching.

---

## 13. Anti-Bias & Skeptic Constraints

1. **Blind First**: Discover structural patterns before exposing proposed translation labels.
2. **No Cherry-Picking**: Hypotheses must be tested against the complete eligible corpus.
3. **Negative Evidence**: Formally record expected-but-absent forms.
4. **Holdout Test**: 80/20 corpus split (training set vs held-out evaluation set).
5. **Competing Hypotheses**: Always maintain $\ge 2$ competing alternatives.
6. **Monte Carlo Permutation Controls**: Test against Tier 1 (shuffled), Tier 2 (frequency-preserving), and Tier 3 (Markov-preserving) null corpora.
7. **Semantic Blinding**: Never feed desired translations back into phonetic inference.
8. **Double-Entry Evidence Guard**: A single observation cannot count twice as independent validation.

---

## 14. Phaistos Disk Firewall

* `CORPUS_LINEAR_A` and `CORPUS_PHAISTOS_DISK` remain physically and logically isolated.
* Never transfer phonetic values or meanings between scripts without independent paleographic and statistical proof.
* Phaistos Disc serves as an external test corpus, not proof of Linear A hypotheses.

---

## 15. The 20-Step Discovery Pipeline

```
1. Ingest Canonical Corpus (GORILA / SigLA / Younger)
2. Normalize Sign Identifiers (AB01-AB87, A301-A364)
3. Preserve Paleographic Uncertainty (Confidence vectors)
4. Segment Tokens & Identify Dividers
5. Catalog Numerals & Logograms
6. Generate Prior Phonetic Lattice (Linear B priors as E2)
7. Build Distributional & N-Gram Models
8. Detect Recurring Sequences & Collocations
9. Infer Candidate Morphology (Prefix/Suffix sieves)
10. Exploit Numerical Constraints & Fractional Math (Ferrara 2020)
11. Cluster Contextual & Administrative Environments
12. Test Regional Variations & Scribe Profiles
13. Generate Competing Linguistic Hypotheses
14. Score Hypotheses with Quantitative Metric S
15. Execute Monte Carlo Null Permutations (Fedora PC Batch)
16. Run Corpus Holdout Generalization Tests
17. Evaluate Predictions on Unseen Inscriptions
18. Run Autonomous Skeptic Adversary Gauntlet
19. Multi-Model Consensus (Ollama Local + Cloud Gemini)
20. Final Epistemic Report & Interactive Workbench Visualization
```
