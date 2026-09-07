"""Audit canonical historical decipherment proposals against the Tripartite Blind Skeptic Jury."""

import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from linear_a.llm.jury import SkepticJury

console = Console()

PROPOSALS = [
    {
        "id": "PROP-001",
        "name": "Cyrus Gordon (1966) / Jan Best (1988) — Northwest Semitic",
        "claim": "Linear A is Northwest Semitic: KU-RO = kullu ('all'), KI-RO = killu ('deficit'), PA-DE = Ba'al, JA-SA-SA-RA-ME = Asherah",
        "domain": "lexical / comparative Semitic",
        "assumptions": 4.0,
        "exceptions": 3.0,
        "phonetic_deviations": 3.0,
        "semantic_leaps": 4.0,
        "ad_hoc_rules": 3.0,
    },
    {
        "id": "PROP-002",
        "name": "Leonard Palmer (1958) / Margalit Finkelberg (1990) — Anatolian Luwian",
        "claim": "Linear A is Anatolian Luwian: A-SA-SA-RA-ME = *asassara-mis ('my lady'), TA-NA-I-*301-TI = *tani-ti ('she placed'), KU-RO = kula ('total')",
        "domain": "morphological / comparative Indo-European",
        "assumptions": 2.0,
        "exceptions": 2.0,
        "phonetic_deviations": 2.0,
        "semantic_leaps": 2.0,
        "ad_hoc_rules": 2.0,
    },
    {
        "id": "PROP-003",
        "name": "Yves Duhoux (1978) / Gareth Owens (2007) — Minoan Pre-Hellenic Language Isolate",
        "claim": "Linear A represents an indigenous Aegean agglutinative isolate with JA-/A- nominal prefixing, allative -TE suffixation, and open CV phonotactics",
        "domain": "systemic morphology / structural isolate",
        "assumptions": 1.0,
        "exceptions": 1.0,
        "phonetic_deviations": 0.0,
        "semantic_leaps": 0.0,
        "ad_hoc_rules": 1.0,
    },
    {
        "id": "PROP-004",
        "name": "David Packard (1974) — Non-Greek Agglutinative Syllabary",
        "claim": "Statistical information theory demonstrates Linear A has open CV syllable structure without Greek consonant clusters or nominal declension",
        "domain": "statistical information theory / phonotactics",
        "assumptions": 0.0,
        "exceptions": 0.0,
        "phonetic_deviations": 0.0,
        "semantic_leaps": 0.0,
        "ad_hoc_rules": 0.0,
    },
]


def run_audit_suite():
    jury = SkepticJury()
    results = []

    console.print(Panel("[bold cyan]Linear A Decipherment Gauntlet: Canonical Historical Proposals[/bold cyan]\n[dim]Tripartite Blind Skeptic Jury (Qwen 2.5 7B, Gemma 2 9B, Llama 3.2 3B)[/dim]"))

    for p in PROPOSALS:
        console.print(f"\n[bold yellow]Auditing {p['id']}: {p['name']}[/bold yellow]")
        dossier = jury.evaluate_claim(
            claim=p["claim"],
            claim_domain=p["domain"],
            arbitrary_assumptions=p["assumptions"],
            exceptions=p["exceptions"],
            phonetic_deviations=p["phonetic_deviations"],
            semantic_leaps=p["semantic_leaps"],
            ad_hoc_rules=p["ad_hoc_rules"],
        )

        results.append({
            "id": p["id"],
            "name": p["name"],
            "claim": p["claim"],
            "domain": p["domain"],
            "score": dossier.quantitative_score_s,
            "grade": dossier.epistemic_grade,
            "verdict": dossier.unanimous_verdict,
            "summary": dossier.synthesis_summary,
            "jurors": [
                {
                    "model": c.model_name,
                    "persona": c.persona,
                    "falsified": c.is_falsified,
                    "penalty": c.score_penalty,
                    "critique": c.critique_text,
                }
                for c in dossier.juror_critiques
            ],
        })

        console.print(f"• Score S: [bold cyan]{dossier.quantitative_score_s:.1f}/100[/bold cyan] ({dossier.epistemic_grade}) | Verdict: [bold]{dossier.unanimous_verdict}[/bold]")

    # Save outputs
    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "jury_proposals_dossier.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Summary
    md_path = Path("reports/decipherment_proposals_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tripartite Blind Skeptic Jury — Decipherment Proposals Audit\n\n")
        f.write("| Proposal ID | Hypothesis & Proponent | Domain | Quantitative Score S | Epistemic Grade | Consensus Verdict |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| **{r['id']}** | {r['name']} | `{r['domain']}` | **{r['score']:.1f} / 100** | {r['grade']} | {r['verdict']} |\n")
        f.write("\n---\n\n")
        for r in results:
            f.write(f"## {r['id']}: {r['name']}\n\n")
            f.write(f"- **Claim**: {r['claim']}\n")
            f.write(f"- **Score S**: **{r['score']:.1f}/100** ({r['grade']})\n")
            f.write(f"- **Consensus Verdict**: `{r['verdict']}`\n\n")
            f.write("### Juror Critiques\n\n")
            for j in r["jurors"]:
                f.write(f"#### {j['persona']} (`{j['model']}`)\n")
                f.write(f"- **Falsified**: {j['falsified']} (Penalty: {j['penalty']:.1f})\n")
                f.write(f"> {j['critique']}\n\n")

    console.print(f"\n[bold green]✓ Audit Suite Complete![/bold green] Saved to [cyan]{json_path}[/cyan] and [cyan]{md_path}[/cyan]")


if __name__ == "__main__":
    run_audit_suite()
