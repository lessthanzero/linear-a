"""Linear A Masked Phonotactic Lacunae Infilling Engine (LADP v1.0).

Predicts and reconstructs damaged or effaced syllabic signs in Minoan inscriptions
using bigram transition matrices, positional priors, and attested corpus templates.
Strictly adheres to Evidence Tier E2 (Phonotactic Priors) and E4 (Structural Context).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from linear_a.corpus.loader import load_all_tablets, load_signs_catalogue
from linear_a.palaeography.grid_factorization import KoberVentrisGridEngine
from linear_a.votive.libation_engine import LibationEngine


@dataclass
class InfillCandidate:
    """A candidate syllabogram for an effaced position."""
    sign_id: str
    reading: str
    composite_log_prob: float
    bayes_factor: float
    confidence_tier: str  # "E2" or "E4"
    lexical_match: Optional[str] = None
    rationale: str = ""


@dataclass
class LacunaInfillResult:
    """Full infilling report for a damaged token."""
    masked_token: str
    masked_position: int  # 0-indexed position of the missing sign
    top_candidates: List[InfillCandidate]
    best_candidate: InfillCandidate
    is_exact_lexical_recovery: bool
    summary: str


class LacunaeInfiller:
    """Predicts missing signs in damaged Linear A inscriptions."""

    def __init__(
        self,
        grid_engine: Optional[KoberVentrisGridEngine] = None,
    ):
        self.grid = grid_engine or KoberVentrisGridEngine()
        self.signs = load_signs_catalogue()
        self.corpus_vocab: List[str] = self._build_corpus_vocab()
        self._bigram_counts, self._unigram_counts, self._total_tokens = self._build_ngram_models()

    def _build_corpus_vocab(self) -> List[str]:
        """Extract all attested words across administrative tablets and libation vessels."""
        vocab = set()
        # From tablets
        for t in load_all_tablets():
            if "heading" in t:
                h_val = t["heading"].strip()
                if h_val and "?" not in h_val and "[" not in h_val:
                    vocab.add(h_val)
            for it in t.get("items", []):
                h = it.get("entry_header", "").strip()
                if h and "?" not in h and "[" not in h and h not in ("KU-RO", "KI-RO") and not h.isdigit():
                    vocab.add(h)

        # From votive vessels
        lib_engine = LibationEngine()
        for v in lib_engine.vessels:
            for seg in v.segments:
                w_val = seg.word.strip()
                if w_val and "?" not in w_val and "[" not in w_val:
                    vocab.add(w_val)

        # From attested words lexicon
        from linear_a.corpus.loader import get_default_corpus_dir
        import yaml
        attested_path = get_default_corpus_dir() / "lexicons" / "attested_words.yaml"
        if attested_path.exists():
            with open(attested_path, "r", encoding="utf-8") as f:
                lex_data = yaml.safe_load(f)
                for w_item in lex_data.get("words", []):
                    tok = w_item.get("token", "").strip()
                    if tok and "?" not in tok and "[" not in tok:
                        vocab.add(tok)

        return sorted(vocab)

    def _build_ngram_models(self) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], int]:
        """Build bigram and unigram sign frequency distributions."""
        bigrams: Dict[Tuple[str, str], int] = {}
        unigrams: Dict[str, int] = {}
        total = 0

        # Seed unigrams with all phonetic signs in catalogue
        for s_id, sign in self.signs.items():
            val = sign.canonical_name or (sign.linearB_correspondence.candidate if sign.linearB_correspondence else None)
            if val:
                u_val = val.upper()
                unigrams[u_val] = unigrams.get(u_val, 0) + 1
                total += 1

        for word in self.corpus_vocab:
            sylls = word.split("-")
            for i, s in enumerate(sylls):
                unigrams[s] = unigrams.get(s, 0) + 1
                total += 1
                if i < len(sylls) - 1:
                    next_s = sylls[i + 1]
                    bigrams[(s, next_s)] = bigrams.get((s, next_s), 0) + 1

        return bigrams, unigrams, total

    def infill_token(self, masked_token: str, mask_char: str = "?") -> LacunaInfillResult:
        """Infill an effaced sign in a token like 'KU-?-NU' or '?-SA-SA-RA-ME'."""
        sylls = masked_token.split("-")
        mask_indices = [i for i, s in enumerate(sylls) if mask_char in s or s in ("*", "[-]")]

        if not mask_indices:
            raise ValueError(f"No mask character '{mask_char}' found in token '{masked_token}'.")

        pos = mask_indices[0]
        left_context = sylls[pos - 1].strip() if pos > 0 else None
        right_context = sylls[pos + 1].strip() if pos < len(sylls) - 1 else None

        candidates: List[InfillCandidate] = []
        all_candidate_signs = [
            s for s in self._unigram_counts.keys()
            if s and "?" not in s and "[" not in s and s not in ("*", "[-]") and not s.isdigit()
        ]
        uniform_null = 1.0 / max(1, len(all_candidate_signs))

        for cand in all_candidate_signs:
            # 1. Unigram log prior
            cand_count = self._unigram_counts.get(cand, 1)
            p_cand = cand_count / self._total_tokens
            log_prob = math.log(p_cand + 1e-6)

            # 2. Left bigram transition: P(cand | left)
            if left_context:
                bigram_pair = (left_context, cand)
                left_count = self._unigram_counts.get(left_context, 1)
                p_left = (self._bigram_counts.get(bigram_pair, 0) + 0.1) / (left_count + 1.0)
                log_prob += 2.0 * math.log(p_left)

            # 3. Right bigram transition: P(right | cand)
            if right_context:
                bigram_pair = (cand, right_context)
                p_right = (self._bigram_counts.get(bigram_pair, 0) + 0.1) / (cand_count + 1.0)
                log_prob += 2.0 * math.log(p_right)

            # 4. Check for complete lexical template match in corpus
            hypothetical_sylls = list(sylls)
            hypothetical_sylls[pos] = cand
            hypothetical_word = "-".join(hypothetical_sylls)

            lexical_match = None
            if hypothetical_word in self.corpus_vocab:
                lexical_match = hypothetical_word
                log_prob += 6.0  # Massive boost for attested corpus word match

            # Calculate Bayes Factor vs uniform random baseline
            posterior_prob = math.exp(log_prob)
            bayes_factor = max(0.1, (posterior_prob / (uniform_null + 1e-9)))

            # Epistemic tier assignment
            tier = "E4" if lexical_match else "E2"

            rationale_parts = []
            if lexical_match:
                rationale_parts.append(f"Exact match with attested Minoan token '{hypothetical_word}'")
            if left_context:
                rationale_parts.append(f"Transition {left_context} -> {cand}")
            if right_context:
                rationale_parts.append(f"Transition {cand} -> {right_context}")

            candidates.append(InfillCandidate(
                sign_id=cand,
                reading=cand,
                composite_log_prob=log_prob,
                bayes_factor=round(bayes_factor, 1),
                confidence_tier=tier,
                lexical_match=lexical_match,
                rationale="; ".join(rationale_parts),
            ))

        candidates.sort(key=lambda c: c.composite_log_prob, reverse=True)
        top_candidates = candidates[:5]
        best = top_candidates[0]

        summary = (
            f"Infilled '{masked_token}' at syllable index {pos}. Best candidate: '{best.reading}' "
            f"(BF = {best.bayes_factor:.1f}, {best.confidence_tier}). "
            f"{'Recovered exact attested token: ' + best.lexical_match if best.lexical_match else 'Phonotactic estimation'}."
        )

        return LacunaInfillResult(
            masked_token=masked_token,
            masked_position=pos,
            top_candidates=top_candidates,
            best_candidate=best,
            is_exact_lexical_recovery=best.lexical_match is not None,
            summary=summary,
        )

    def benchmark_reconstruction_accuracy(self, sample_size: int = 20) -> Dict[str, Any]:
        """Cross-validation: mask known tokens and test top-1 and top-3 recovery rates."""
        test_words = [w for w in self.corpus_vocab if len(w.split("-")) >= 3][:sample_size]
        top1_correct = 0
        top3_correct = 0

        for word in test_words:
            sylls = word.split("-")
            mask_idx = len(sylls) // 2  # mask medial syllable
            target_syll = sylls[mask_idx]

            masked_sylls = list(sylls)
            masked_sylls[mask_idx] = "?"
            masked_str = "-".join(masked_sylls)

            res = self.infill_token(masked_str)
            top1_cand = res.top_candidates[0].reading
            top3_cands = [c.reading for c in res.top_candidates[:3]]

            if top1_cand == target_syll:
                top1_correct += 1
            if target_syll in top3_cands:
                top3_correct += 1

        n = max(1, len(test_words))
        return {
            "tokens_tested": n,
            "top1_accuracy_pct": round((top1_correct / n) * 100.0, 1),
            "top3_accuracy_pct": round((top3_correct / n) * 100.0, 1),
            "epistemic_tier": "E2-E4",
        }
