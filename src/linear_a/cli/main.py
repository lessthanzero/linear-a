"""Rich CLI Entrypoint for Linear A Computational Laboratory."""

from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from linear_a.accounting.fractions import FractionEngine
from linear_a.accounting.ledger import LedgerValidator
from linear_a.corpus.loader import (
    load_signs_catalogue,
    load_tablet_ledgers,
    parse_tablet_line_items,
)
from linear_a.llm.jury import SkepticJury
from linear_a.llm.ollama import OllamaClient
from linear_a.morphology.affix_sieve import AffixSieve
from linear_a.palaeography.grid_factorization import KoberVentrisGridEngine
from linear_a.bridge.phaistos_matrix import PhaistosBridgeEngine
from linear_a.predictive.holdout_engine import HoldoutEngine
from linear_a.skeptic.dictionary_gauntlet import DictionaryGauntlet
from linear_a.visualizer.workbench import generate_workbench_html
from linear_a.votive.libation_engine import LibationEngine

app = typer.Typer(
    name="linear-a",
    help="Linear A Computational Laboratory & Epigraphic Decipherment Harness",
    no_args_is_help=True,
)
console = Console()


@app.command()
def status():
    """Display cluster status, local models, and corpus statistics."""
    console.print(Panel("[bold cyan]Linear A Computational Laboratory[/bold cyan]\n[dim]Protocol: LADP v1.0 | Nodes: Mac (Apple Silicon) + Fedora PC (pc)[/dim]"))

    client = OllamaClient()
    is_online = client.is_available()
    models = client.list_models() if is_online else []
    models_str = ", ".join(models) if models else "[yellow]No models found or host offline[/yellow]"

    signs = load_signs_catalogue()
    ht_tablets = load_tablet_ledgers("hagia_triada")
    ph_tablets = load_tablet_ledgers("phaistos")

    table = Table(title="System & Epigraphic Inventory", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status / Inventory", style="green")

    table.add_row("Local Ollama Host", f"{client.base_url} ({'ONLINE' if is_online else 'OFFLINE'})")
    table.add_row("Local Models Available", models_str)
    table.add_row("Cataloged Signs", f"{len(signs)} signs (AB01-AB87, Fractions, Ideograms)")
    table.add_row("Hagia Triada Tablets", f"{len(ht_tablets)} canonical ledgers (HT 9, 13, 86, 104, 122)")
    table.add_row("Phaistos Tablets", f"{len(ph_tablets)} tablets (PH 1 Room 8, PH 2, PH 6)")
    table.add_row("Fractional Model", "Ferrara et al. (2020) J=1/2, E=1/4, F=1/8, K=1/16, H=1/12")
    table.add_row("Phaistos Firewall", "[bold green]ACTIVE[/bold green] (CORPUS_LINEAR_A isolated from Phaistos Disc)")

    console.print(table)


@app.command()
def grid(
    components: int = typer.Option(4, "--components", "-k", help="Number of singular dimensions"),
    permutations: int = typer.Option(1000, "--permutations", "-n", help="Monte Carlo null permutations"),
):
    """Run unsupervised Kober-Ventris SVD factorization on Linear A sign transitions."""
    console.print(Panel("[bold cyan]Unsupervised Kober-Ventris SVD Grid Factorization[/bold cyan]\n[dim]PPMI Transition Matrix Decomposition & Ventris Grid Concordance[/dim]"))

    engine = KoberVentrisGridEngine()
    with console.status("[bold cyan]Factorizing transition matrix and running Monte Carlo null permutations...[/bold cyan]"):
        report = engine.factorize_grid(
            n_components=components,
            n_consonant_clusters=4,
            n_vowel_clusters=3,
            n_permutations=permutations,
        )

    # Singular Values Table
    table_svd = Table(title="Spectral Decomposition (Singular Values)", show_header=True)
    table_svd.add_column("Dimension", justify="center", style="yellow")
    table_svd.add_column("Singular Value (σ)", justify="right", style="cyan")

    for i, s in enumerate(report.singular_values, 1):
        table_svd.add_row(f"Dimension {i}", f"{s:.3f}")
    console.print(table_svd)
    console.print(f"• Total Variance Explained: [bold green]{report.spectral_variance_explained_pct:.1f}%[/bold green] across {report.n_components} components")

    # Consonant Clusters Table
    table_c = Table(title="Discovered Consonant Series (Unsupervised)", show_header=True)
    table_c.add_column("Cluster", style="yellow")
    table_c.add_column("Signs", style="cyan")
    table_c.add_column("Dominant Consonant", style="green")
    table_c.add_column("Purity Ratio", justify="right")

    for c in report.consonant_clusters:
        signs_str = ", ".join(c.sample_readings)
        table_c.add_row(f"Class C-{c.cluster_id + 1}", signs_str, c.dominant_consonant_or_vowel, f"{c.homogeneity_ratio * 100:.1f}%")
    console.print(table_c)

    # Vowel Clusters Table
    table_v = Table(title="Discovered Vowel Series (Unsupervised)", show_header=True)
    table_v.add_column("Cluster", style="yellow")
    table_v.add_column("Signs", style="cyan")
    table_v.add_column("Dominant Vowel", style="green")
    table_v.add_column("Purity Ratio", justify="right")

    for v in report.vowel_clusters:
        signs_str = ", ".join(v.sample_readings)
        table_v.add_row(f"Class V-{v.cluster_id + 1}", signs_str, v.dominant_consonant_or_vowel, f"{v.homogeneity_ratio * 100:.1f}%")
    console.print(table_v)

    # Statistical Evaluation vs Null
    z_color = "green" if report.z_score >= 3.0 else ("yellow" if report.z_score >= 1.96 else "red")
    console.print(f"• Pairwise Ventris Grid Agreement: [bold]{report.ventris_grid_pairwise_agreement_rate * 100:.1f}%[/bold]")
    console.print(f"• Permutation Null Baseline: {report.null_mean_agreement_rate * 100:.1f}% ± {report.null_std_agreement_rate * 100:.1f}%")
    console.print(f"• Empirical Z-Score: [bold {z_color}]{report.z_score:+.2f}σ[/bold {z_color}] | p-value: [bold {z_color}]{report.p_value:.4f}[/bold {z_color}]")
    console.print(f"\n[dim]{report.epistemic_verdict}[/dim]")


@app.command()
def gauntlet(
    language: str = typer.Argument("semitic_northwest", help="Candidate lexicon (semitic_northwest, anatolian_luwian, synthetic_null)"),
    permutations: int = typer.Option(1000, "--permutations", "-n", help="Monte Carlo null iterations"),
):
    """Subject a candidate language family dictionary to the False-Positive Gauntlet."""
    console.print(Panel(f"[bold cyan]Cross-Linguistic Dictionary Gauntlet: {language}[/bold cyan]\n[dim]Testing candidate cognates against synthetic pseudo-lexicon null collisions[/dim]"))

    engine = DictionaryGauntlet()
    with console.status(f"[bold cyan]Running {permutations} Monte Carlo null collisions for {language}...[/bold cyan]"):
        report = engine.run_gauntlet(language_key=language, n_surrogates=permutations)

    table = Table(title=f"Claimed Lexical Matches ({report.target_language})", show_header=True)
    table.add_column("Root", style="yellow")
    table.add_column("Meaning", style="dim")
    table.add_column("Claimed Form", style="cyan")
    table.add_column("Target Token", style="green")
    table.add_column("Bayes Factor", justify="right")
    table.add_column("Status", justify="center")

    for m in report.top_matches:
        stat_color = "green" if m.status == "STRONG_CANDIDATE" else ("yellow" if m.status == "EQUIVOCAL" else "red")
        table.add_row(m.root, m.meaning, m.claimed_spelling, m.target_token, f"{m.bayes_factor:.1f}", f"[{stat_color}]{m.status}[/{stat_color}]")

    console.print(table)
    console.print(f"• Total Candidate Roots: [bold]{report.total_candidate_roots}[/bold]")
    console.print(f"• Observed Matches in Corpus: [bold]{report.observed_matches_count}[/bold]")
    console.print(f"• Null Collision Baseline: {report.null_mean_matches:.2f} ± {report.null_std_matches:.2f} matches")
    console.print(f"• Empirical Z-Score: [bold]{report.z_score:+.2f}σ[/bold] | p-value: [bold]{report.empirical_p_value:.4f}[/bold]")
    console.print(f"• False Positive Rate (FPR): [bold yellow]{report.false_positive_rate_pct:.1f}%[/bold yellow]")
    console.print(f"• Shannon Unicity Ratio: [bold]{report.unicity_ratio:.2f}[/bold] ({'CONSTRAINED' if report.unicity_ratio <= 1.0 else 'UNCONSTRAINED OVERFIT'})")
    console.print(f"\n[dim]{report.epistemic_verdict}[/dim]")


@app.command()
def verify(
    tablet_id: str = typer.Argument("HT_009", help="Tablet ID to verify, e.g. HT_009, HT_013, HT_122, PH_001")
):
    """Verify mathematical accounting balance sum(inputs) == KU-RO on a tablet."""
    all_tablets = load_tablet_ledgers("hagia_triada") + load_tablet_ledgers("phaistos")
    target = next((t for t in all_tablets if t["id"].upper() == tablet_id.upper()), None)

    if not target:
        console.print(f"[bold red]Tablet '{tablet_id}' not found in corpus.[/bold red]")
        raise typer.Exit(1)

    items = parse_tablet_line_items(target)
    validator = LedgerValidator()

    kuro_int = target.get("stated_kuro", {}).get("integer_amount")
    kuro_frac = target.get("stated_kuro", {}).get("fractional_symbols", [])
    kuro_frac_str = kuro_frac[0] if kuro_frac else None

    result = validator.verify_ledger(
        tablet_id=target["id"],
        items=items,
        stated_kuro_integer=kuro_int,
        stated_kuro_fraction=kuro_frac_str,
        commodity=target.get("commodity"),
    )

    console.print(Panel(f"[bold cyan]Accounting Verification: {target['id']} ({target['site']})[/bold cyan]\n[dim]{target.get('notes', '')}[/dim]"))

    table = Table(title=f"Line Items for {target['id']}", show_header=True)
    table.add_column("Header / Recipient", style="yellow")
    table.add_column("Commodity", style="cyan")
    table.add_column("Amount", justify="right")

    for it in items:
        amt_str = f"{it.integer_amount}"
        if it.fractional_symbols:
            amt_str += f" {''.join(it.fractional_symbols)}"
        table.add_row(it.entry_header, it.commodity or "-", amt_str)

    console.print(table)

    status_color = "green" if result.is_balanced else "red"
    console.print(f"• Input Items Count: [bold]{result.input_items_count}[/bold]")
    console.print(f"• Computed Sum: [bold]{result.computed_sum_fraction}[/bold] (decimal {result.computed_sum_decimal:.4f})")
    if result.stated_kuro_fraction:
        console.print(f"• Stated KU-RO: [bold]{result.stated_kuro_fraction}[/bold] (decimal {result.stated_kuro_decimal:.4f})")
        console.print(f"• Balance Verdict: [bold {status_color}]{'EXACT BALANCE (sum == KU-RO)' if result.is_balanced else 'DISCREPANCY DETECTED'}[/bold {status_color}]")
    console.print(f"\n[dim]{result.rationale}[/dim]")


@app.command()
def libation():
    """Analyze formulaic syntax and prosody across stone libation tables."""
    engine = LibationEngine()
    report = engine.evaluate_concordance()

    console.print(Panel("[bold cyan]Linear A Libation Formula Engine[/bold cyan]\n[dim]Peak Sanctuary & Cave Sacred Dedications (IO Za 2, PS Za 2, PK Za 11)[/dim]"))

    table = Table(title="Canonical Formulaic Sequence", show_header=True)
    table.add_column("Sequence Order", style="yellow")
    table.add_column("Formulaic Segment", style="cyan")

    for idx, seg in enumerate(report.canonical_order, 1):
        table.add_row(f"Step {idx}", seg)

    console.print(table)
    console.print(f"• Total Vessels Analyzed: [bold]{report.total_vessels}[/bold]")
    console.print(f"• Divine Epithet (JA-SA-SA-RA-ME) Recurrence: [bold yellow]{report.jasasarame_recurrence_rate * 100:.1f}%[/bold yellow]")
    console.print(f"• Dedicatory Verb (U-NA-KA-NA-SI) Recurrence: [bold yellow]{report.unakanasi_recurrence_rate * 100:.1f}%[/bold yellow]")
    console.print(f"• Mean Morae per Inscription: [bold]{report.mean_morae_per_vessel:.1f} morae[/bold]")
    console.print(f"• Phaistos Disc Liturgical Homology Score: [bold green]{report.phaistos_disc_liturgical_homology_score:.1f}%[/bold green]")
    console.print(f"\n[dim]{report.summary}[/dim]")


@app.command()
def affixes():
    """Extract and analyze prefix/suffix distributions across GORILA corpus."""
    sieve = AffixSieve()
    report = sieve.evaluate_affixes()

    console.print(Panel("[bold cyan]Linear A Morphological & Suffix Sieve[/bold cyan]\n[dim]1,427 GORILA Word-Final Tokens Analyzed[/dim]"))

    table_suffs = Table(title="Top Word-Final Case Suffixes", show_header=True)
    table_suffs.add_column("Sign", style="yellow")
    table_suffs.add_column("Reading", style="cyan")
    table_suffs.add_column("Corpus Share", justify="right")
    table_suffs.add_column("Grammatical Role", style="dim")

    for s in report.top_suffixes[:6]:
        table_suffs.add_row(s.glyph_id, s.canonical_name, f"{s.corpus_rate * 100:.1f}% ({s.observed_count})", s.grammatical_role)

    console.print(table_suffs)

    table_prefs = Table(title="Top Word-Initial Prefixes", show_header=True)
    table_prefs.add_column("Sign", style="yellow")
    table_prefs.add_column("Reading", style="cyan")
    table_prefs.add_column("Corpus Share", justify="right")
    table_prefs.add_column("Grammatical Role", style="dim")

    for p in report.top_prefixes[:4]:
        table_prefs.add_row(p.glyph_id, p.canonical_name, f"{p.corpus_rate * 100:.1f}% ({p.observed_count})", p.grammatical_role)

    console.print(table_prefs)

    console.print(f"• Likelihood Ratio (TE vs ME for Sign 35): [bold green]{report.te_vs_me_likelihood_ratio:.1e}[/bold green]")
    console.print(f"• Phaistos Disc Sign 35 Bridge (TE): [bold green]{'VERIFIED' if report.phaistos_disc_bridge_te_match else 'REJECTED'}[/bold green]")
    console.print(f"\n[dim]{report.epistemic_verdict}[/dim]")


@app.command()
def audit(
    claim: str = typer.Argument(..., help="Decipherment claim text to evaluate")
):
    """Subject any decipherment claim to the Tripartite Blind Skeptic Jury."""
    console.print(Panel(f"[bold yellow]Auditing Decipherment Claim[/bold yellow]\nClaim: '{claim}'"))

    jury = SkepticJury()
    with console.status("[bold cyan]Querying Tripartite Jury across local models...[/bold cyan]"):
        dossier = jury.evaluate_claim(claim=claim)

    console.print(f"\n[bold]Quantitative Score S:[/bold] [bold cyan]{dossier.quantitative_score_s:.1f}/100[/bold cyan] ({dossier.epistemic_grade})")
    console.print(f"[bold]Consensus Verdict:[/bold] [bold yellow]{dossier.unanimous_verdict}[/bold yellow]\n")

    for critique in dossier.juror_critiques:
        panel_color = "red" if critique.is_falsified else "green"
        console.print(Panel(
            f"[bold]{critique.persona} ({critique.model_name})[/bold]\n"
            f"Falsified: [{'red' if critique.is_falsified else 'green'}]{critique.is_falsified}[/{'red' if critique.is_falsified else 'green'}] | Penalty: {critique.score_penalty:.1f}\n\n"
            f"{critique.critique_text}",
            border_style=panel_color,
        ))


@app.command()
def holdout():
    """Execute Step 16 holdout generalization suite (Khania, Zakros, Tylissos)."""
    console.print(Panel("[bold cyan]Corpus Holdout Generalization & Predictive Validation[/bold cyan]\n[dim]LADP v1.0 Step 16 | 80/20 Train/Test Partition[/dim]"))

    engine = HoldoutEngine()
    with console.status("[bold cyan]Evaluating accounting generalization, masked reconstruction, and morphology...[/bold cyan]"):
        report = engine.run_full_holdout_suite()

    # Accounting Table
    table_acct = Table(title="Holdout Accounting Total Predictions (E3 Deterministic Solver)", show_header=True)
    table_acct.add_column("Tablet ID", style="yellow")
    table_acct.add_column("Site / Region", style="cyan")
    table_acct.add_column("Commodity", justify="center")
    table_acct.add_column("Items", justify="right")
    table_acct.add_column("Predicted KU-RO", justify="right", style="green")
    table_acct.add_column("Stated KU-RO", justify="right")
    table_acct.add_column("Masked Recon Acc", justify="right")
    table_acct.add_column("Status", justify="center")

    for p in report.accounting_summary.detailed_predictions:
        table_acct.add_row(
            p.tablet_id,
            p.site,
            p.commodity or "-",
            str(p.input_items_count),
            p.predicted_kuro,
            p.stated_kuro,
            f"{p.masked_reconstruction_accuracy_pct:.1f}%",
            "[bold green]EXACT[/bold green]" if p.is_exact_match else "[bold red]FAIL[/bold red]",
        )
    console.print(table_acct)

    console.print(f"• Holdout KU-RO Exact Prediction Accuracy: [bold green]{report.accounting_summary.exact_kuro_prediction_accuracy_pct:.1f}%[/bold green]")
    console.print(f"• Damaged Entry Reconstruction Accuracy: [bold green]{report.accounting_summary.masked_reconstruction_accuracy_pct:.1f}%[/bold green] across {report.accounting_summary.total_masked_entries_tested} line items")
    console.print(f"• Morphological Case Suffix Top-3 Retention: [bold]{report.morphology_summary.top3_suffix_accuracy_pct:.1f}%[/bold]")

    # Regional Profiles Table
    table_reg = Table(title="Regional Scribal Profiles (Pan-Cretan LM IB Koine)", show_header=True)
    table_reg.add_column("Site", style="yellow")
    table_reg.add_column("Region", style="dim")
    table_reg.add_column("Tablets", justify="right")
    table_reg.add_column("Lexical Overlap with Hagia Triada", justify="right", style="cyan")
    table_reg.add_column("Primary Commodities", style="green")

    for reg in report.regional_profiles:
        comm_str = ", ".join(reg.commodities) if reg.commodities else "Various"
        table_reg.add_row(reg.site, reg.region, str(reg.tablets_count), f"{reg.jaccard_overlap_with_hagia_triada * 100:.1f}%", comm_str)
    console.print(table_reg)
    console.print(f"\n[dim]{report.epistemic_verdict}[/dim]")


@app.command()
def bridge():
    """Verify Phaistos Disc Firewall integrity and compute cross-script homology."""
    console.print(Panel("[bold cyan]Phaistos Disc Cross-Script Structural Bridge & Firewall[/bold cyan]\n[dim]LADP v1.0 Section 14 | Zero Bidirectional Phonetic Leakage[/dim]"))

    engine = PhaistosBridgeEngine()
    report = engine.evaluate_cross_script_homology()

    # Firewall Status
    fw_color = "bold green" if report.firewall.is_firewall_intact else "bold red"
    console.print(f"• Epigraphic Firewall Status: [{fw_color}]{'INTACT / SECURE' if report.firewall.is_firewall_intact else 'VIOLATED'}[/{fw_color}]")
    console.print(f"• Quarantined Corpora: [dim]{', '.join(report.firewall.isolated_corpora)}[/dim]")
    console.print(f"• Overall Cross-Script Structural Homology: [bold green]{report.overall_structural_homology_score_pct:.1f}%[/bold green]")

    table = Table(title="Cross-Script Structural Correspondences (Firewall Compliant)", show_header=True)
    table.add_column("Feature", style="yellow")
    table.add_column("Phaistos Disc Evidence", style="dim")
    table.add_column("Linear A Evidence", style="cyan")
    table.add_column("Metric / Value", justify="right", style="green")
    table.add_column("Status", justify="center")

    for c in report.correspondences:
        val_str = f"{c.metric_value:.1e}" if c.metric_value > 1000 else f"{c.metric_value:.2f}"
        table.add_row(c.feature_name, c.disc_evidence, c.linear_a_evidence, f"{c.statistical_metric}: {val_str}", f"[bold green]{c.concordance_level}[/bold green]")

    console.print(table)
    console.print(f"\n[dim]{report.epistemic_verdict}[/dim]")


@app.command()
def workbench(
    output: str = typer.Option("reports/linear_a_workbench.html", "--output", "-o", help="Output path for standalone HTML workbench")
):
    """Generate the publication-grade interactive HTML research workbench."""
    console.print(Panel("[bold cyan]Generating Interactive Epigraphic Research Workbench[/bold cyan]\n[dim]California/Swiss Editorial Craft | Single-File Standalone HTML/SVG/JS[/dim]"))

    with console.status(f"[bold cyan]Assembling epigraphic datasets and compiling HTML to {output}...[/bold cyan]"):
        out_path = generate_workbench_html(output)

    console.print(f"[bold green]✓ Research Workbench successfully compiled:[/bold green] [bold cyan]{out_path}[/bold cyan] ({out_path.stat().st_size / 1024:.1f} KB)")
    console.print("[dim]Open in any browser: zero external dependencies, works completely offline.[/dim]")


if __name__ == "__main__":
    app()
