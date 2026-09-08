"""Tests for Epigrapher Peer-Review Adjudication Portal Engine."""

from linear_a.predictive.adjudication_portal import (
    VALID_VERDICTS,
    AdjudicationPortalEngine,
    AdjudicationPortalReport,
    ReviewPacket,
)


def test_compile_review_packets():
    engine = AdjudicationPortalEngine()
    assert len(engine.packets) >= 20

    first = engine.packets[0]
    assert isinstance(first, ReviewPacket)
    assert first.token_id
    assert first.document_id
    assert first.proposed_sign
    assert first.evidence_tier in ("E1", "E3", "E4", "E5")
    assert len(first.allowed_verdicts) == 4


def test_validate_verdict():
    engine = AdjudicationPortalEngine()

    valid_record = {
        "token_id": "HT_085_tok_1",
        "reviewer_name": "Dr. E. Epigrapher",
        "reviewer_institution": "University of Cambridge",
        "verdict": "CONFIRM_RESTORATION",
        "citation_source": "GORILA I: 130; Godart & Olivier 1976",
        "scholarly_rationale": "Traces of horizontal stroke match sign PA.",
    }
    is_valid, err = engine.validate_verdict(valid_record)
    assert is_valid is True
    assert err is None

    # Invalid: missing citation
    no_cit = dict(valid_record)
    no_cit["citation_source"] = ""
    is_valid, err = engine.validate_verdict(no_cit)
    assert is_valid is False
    assert "citation" in err.lower()

    # Invalid: unknown verdict
    bad_v = dict(valid_record)
    bad_v["verdict"] = "SPECULATIVE_GUESS"
    is_valid, err = engine.validate_verdict(bad_v)
    assert is_valid is False
    assert "Invalid verdict" in err


def test_export_adjudications_yaml(tmp_path):
    engine = AdjudicationPortalEngine()
    out_file = tmp_path / "adjudications_test.yaml"

    verdicts = [
        {
            "token_id": "HT_085_tok_1",
            "reviewer_name": "Dr. E. Epigrapher",
            "reviewer_institution": "University of Cambridge",
            "verdict": "CONFIRM_RESTORATION",
            "citation_source": "GORILA I: 130; Godart & Olivier 1976",
            "scholarly_rationale": "Traces match PA (*03).",
        },
        {
            "token_id": "HT_085_tok_3",
            "reviewer_name": "Dr. E. Epigrapher",
            "reviewer_institution": "University of Cambridge",
            "verdict": "REJECT_RESTORATION",
            "citation_source": "GORILA I: 131",
            "scholarly_rationale": "Break is too severe to confirm TA.",
        },
    ]

    res = engine.export_adjudications_yaml(verdicts, out_file)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "HT_085_tok_1" in content
    assert "CONFIRM_RESTORATION" in content
    assert "adjudicated_tokens_count: 2" in content


def test_recompute_benchmark_with_verdicts():
    engine = AdjudicationPortalEngine()

    sample_verdicts = [
        {"verdict": "CONFIRM_RESTORATION"},
        {"verdict": "CONFIRM_RESTORATION"},
        {"verdict": "REJECT_RESTORATION"},
        {"verdict": "SPLIT_TO_OPEN_E2"},
    ]
    res = engine.recompute_benchmark_with_verdicts(sample_verdicts)

    assert res["total_adjudicated"] == 4
    assert res["confirmed"] == 2
    assert res["rejected"] == 1
    assert res["split_to_e2"] == 1
    # Precision = 2 / (2 + 1) = 66.7%
    assert res["adjudicated_precision_pct"] == 66.7
    # Recall = 2 / 4 = 50.0%
    assert res["adjudicated_recall_pct"] == 50.0


def test_generate_portal_report_and_to_dict():
    engine = AdjudicationPortalEngine()
    report = engine.generate_portal_report()

    assert isinstance(report, AdjudicationPortalReport)
    assert report.total_candidate_packets >= 20
    assert "Adjudication Portal prepared" in report.summary

    rep_dict = report.to_dict()
    assert "total_candidate_packets" in rep_dict
    assert "live_benchmark_metrics" in rep_dict
    assert "packets" in rep_dict
    assert len(rep_dict["packets"]) >= 20
