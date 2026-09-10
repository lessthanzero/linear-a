"""Rich CLI Entrypoint for Linear A Computational Laboratory."""

from pathlib import Path
from typing import Optional
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from linear_a.accounting.ledger import LedgerValidator
from linear_a.corpus.loader import (
    get_tablet_by_id,
    load_signs_catalogue,
    load_tablet_ledgers,
    parse_tablet_line_items,
)
from linear_a.llm.jury import SkepticJury
from linear_a.llm.ollama import OllamaClient
from linear_a.morphology.affix_sieve import AffixSieve
from linear_a.network.scribal_graph import ScribalNetworkGraph
from linear_a.palaeography.grid_factorization import KoberVentrisGridEngine
from linear_a.palaeography.ligatures import LigatureEngine
from linear_a.predictive.holdout_engine import HoldoutEngine
from linear_a.predictive.lacunae_infiller import LacunaeInfiller
from linear_a.reading.interlinear import InterlinearReader
from linear_a.bridge.phaistos_matrix import PhaistosBridgeEngine
from linear_a.skeptic.dictionary_gauntlet import DictionaryGauntlet
from linear_a.skeptic.substratum_filter import PublishedPairBenchmark
from linear_a.predictive.toponym_audit import CretanToponymAudit
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
def substratum(
    permutations: int = typer.Option(10_000, "--permutations", "-n", help="Frequency-preserving null draws"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON report path"),
):
    """Evaluate only published Linear A–Linear B lexical comparison pairs."""
    benchmark = PublishedPairBenchmark()
    report = benchmark.run(surrogates=permutations)
    console.print(Panel(
        "[bold cyan]Published Linear A–Linear B Correspondence Benchmark[/bold cyan]\n"
        f"[dim]Qualified lexical pairs: {report.qualified_entries}/{report.target_entries} | "
        f"Shortfall: {report.qualifying_shortfall}[/dim]"
    ))
    if report.qualified_entries:
        console.print(
            f"Exact-form matches: [bold]{report.observed_exact_matches}[/bold] | "
            f"Null mean: {report.null_mean_matches:.4f} | empirical p: {report.empirical_p_value:.6f}"
        )
    else:
        console.print("[yellow]No score was run: no published lexical pairs meet the benchmark's evidence rules.[/yellow]")
    console.print(f"[dim]{report.limitations}[/dim]")
    if export:
        output = Path(export)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Benchmark report written to {output}[/green]")


@app.command(name="toponym-audit")
def toponym_audit(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON report path"),
):
    """Show source-linked Cretan toponym correspondences and disputes."""
    audit = CretanToponymAudit()
    report = audit.report()
    console.print(Panel(
        f"[bold cyan]{report['title']}[/bold cyan]\n"
        f"[dim]Attested correspondences: {report['status_counts']['attested']} | "
        f"Disputed geographic claims: {report['status_counts']['disputed']}[/dim]"
    ))
    table = Table(title="Published Cross-Script Toponym Evidence", show_header=True)
    table.add_column("ID / Status", style="cyan")
    table.add_column("Conventional forms", style="yellow")
    table.add_column("Claim", style="white")
    table.add_column("Source", style="dim")
    for record in report["records"]:
        forms = " / ".join(value for value in (record["linear_a_form"], record["linear_b_form"], record["alphabetic_form"]) if value)
        citation = record["citations"][0]
        table.add_row(
            f"{record['id']}\n{record['status'].upper()}",
            forms or "—",
            record["claim"],
            f"{citation['author']} ({citation['year']}), {citation['locator']}",
        )
    console.print(table)
    console.print(f"[dim]{report['limitations']}[/dim]")
    if export:
        output = Path(export)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Toponym audit written to {output}[/green]")


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
    console.print(f"• Phaistos Disc Liturgical Homology Score: [bold]{report.phaistos_disc_liturgical_homology_score:.1f}%[/bold] [dim](placeholder; exploratory only)[/dim]")
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


@app.command()
def read(
    target_id: str = typer.Argument(..., help="ID of tablet (e.g. HT_009, KN_001) or vessel (e.g. IO_Za_002)")
):
    """Generate structured 5-tier interlinear reading for any tablet or votive vessel."""
    reader = InterlinearReader()
    doc = None

    # Check if target is a tablet
    tablet_dict = get_tablet_by_id(target_id)
    if tablet_dict:
        doc = reader.parse_tablet(tablet_dict)
    else:
        # Check if target is a libation vessel
        lib_engine = LibationEngine()
        for v in lib_engine.vessels:
            if v.id == target_id:
                doc = reader.parse_vessel(v)
                break

    if not doc:
        console.print(f"[bold red]Error:[/bold red] Inscription '{target_id}' not found in corpus.")
        raise typer.Exit(1)

    balance_badge = "[bold green]✓ EXACT BALANCE[/bold green]" if doc.is_mathematically_balanced else "[bold yellow]⚠ OPEN / UNBALANCED[/bold yellow]"
    console.print(Panel(
        f"[bold cyan]Interlinear Epigraphic Reading: {doc.id} ({doc.site})[/bold cyan]\n"
        f"[dim]Genre: {doc.genre} | Carrier: {doc.carrier} | {balance_badge}[/dim]"
    ))

    table = Table(title=f"Epigraphic Transcription: {doc.id}", show_header=True)
    table.add_column("Line", justify="right", style="dim", width=6)
    table.add_column("Syllabic Ductus", style="bold cyan", width=18)
    table.add_column("Category", style="yellow", width=18)
    table.add_column("Morphological Breakdown", style="dim", width=26)
    table.add_column("Administrative / Liturgical Role", style="green")
    table.add_column("Tier", justify="center", style="magenta", width=6)

    for line in doc.lines:
        for t in line.tokens:
            num_str = f" ({t.numerical_val:g})" if t.numerical_val is not None else ""
            num_str += f" [{t.fraction_display}]" if t.fraction_display else ""
            table.add_row(
                f"L{line.line_index + 1}",
                t.transliteration,
                t.category,
                t.morphology_breakdown,
                f"{t.functional_role}{num_str}",
                t.epistemic_tier,
            )

    console.print(table)
    if doc.stated_total is not None:
        console.print(f"\n• Stated KU-RO: [bold green]{doc.stated_total:g}[/bold green] | Computed Sum: [bold green]{doc.calculated_total:g}[/bold green]")
    console.print(f"[dim]{doc.epistemic_summary}[/dim]")


@app.command()
def infill(
    token: str = typer.Argument(..., help="Damaged token with missing sign marked by '?' (e.g. KU-?-NU)")
):
    """Predict missing/effaced syllabogram in a damaged inscription token."""
    console.print(Panel(f"[bold cyan]Masked Phonotactic Lacunae Infilling[/bold cyan]\n[dim]Token: {token}[/dim]"))

    infiller = LacunaeInfiller()
    try:
        res = infiller.infill_token(token)
    except Exception as e:
        console.print(f"[bold red]Infill error:[/bold red] {e}")
        raise typer.Exit(1)

    table = Table(title=f"Ranked Infill Candidates for '{token}'", show_header=True)
    table.add_column("Rank", justify="right", style="dim", width=6)
    table.add_column("Candidate Sign", style="bold green", width=16)
    table.add_column("Bayes Factor", justify="right", style="cyan", width=14)
    table.add_column("Confidence Tier", justify="center", style="magenta", width=10)
    table.add_column("Phonotactic Rationale & Attested Match", style="yellow")

    for idx, c in enumerate(res.top_candidates):
        table.add_row(
            f"#{idx + 1}",
            c.reading,
            f"{c.bayes_factor:.1f}",
            c.confidence_tier,
            c.rationale,
        )

    console.print(table)
    console.print(f"\n[bold green]Result:[/bold green] {res.summary}")


@app.command()
def ligatures():
    """List and decompose Minoan composite ideograms and fractional compounds."""
    console.print(Panel("[bold cyan]Linear A Composite Ideograms & Fractional Ligatures[/bold cyan]\n[dim]GORILA Corpus | Evidence Tier E3[/dim]"))

    engine = LigatureEngine()
    rep = engine.analyze_corpus()

    table = Table(title="Canonical GORILA Ligatures & Compounds", show_header=True)
    table.add_column("Notation", style="bold cyan", width=12)
    table.add_column("Base Commodity", style="yellow", width=16)
    table.add_column("Modifier", style="magenta", width=12)
    table.add_column("Type", style="dim", width=14)
    table.add_column("Attested Sites", style="green")
    table.add_column("Interpretation Hypothesis", style="white")

    for lig in engine.ligatures:
        table.add_row(
            lig.notation,
            f"{lig.base_name} ({lig.base_commodity})",
            f"{lig.modifier_reading} ({lig.modifier_sign})",
            lig.modifier_type,
            ", ".join(lig.findspots),
            lig.interpretation_hypothesis,
        )

    console.print(table)
    console.print(f"\n[dim]{rep.summary}[/dim]")


@app.command()
def network():
    """Analyze bipartite regional economic network of administrators, commodities, and sites."""
    console.print(Panel("[bold cyan]Bipartite Minoan Regional Economic Network[/bold cyan]\n[dim]Cross-Site Administrator and Commodity Topology[/dim]"))

    graph = ScribalNetworkGraph()
    rep = graph.analyze_network()

    table = Table(title="Top Central Administrative Agents (Degree Centrality)", show_header=True)
    table.add_column("Agent / Entity", style="bold cyan", width=18)
    table.add_column("Degree", justify="right", style="green", width=8)
    table.add_column("Attested Sites", style="yellow", width=28)
    table.add_column("Commodities Managed", style="magenta")

    for a in rep.top_central_agents:
        table.add_row(
            a["agent"],
            str(a["degree"]),
            ", ".join(a["sites"]),
            ", ".join(a["commodities"]),
        )

    console.print(table)

    if rep.cross_site_agents:
        table_cross = Table(title="Inter-Palatial Cross-Site Administrators", show_header=True)
        table_cross.add_column("Agent / Entity", style="bold cyan", width=18)
        table_cross.add_column("Sites Count", justify="right", style="green", width=12)
        table_cross.add_column("Linked Regional Sites", style="yellow")
        table_cross.add_column("Commodities Managed", style="magenta")

        for ca in rep.cross_site_agents:
            table_cross.add_row(
                ca["agent"],
                str(ca["sites_count"]),
                ", ".join(ca["sites"]),
                ", ".join(ca["commodities"]),
            )
        console.print("\n", table_cross)

    console.print(f"\n[dim]{rep.summary}[/dim]")


@app.command()
def solve_lacunae(
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre: votive, administrative, toponymic, arithmetic")
):
    """Run multilateral joint Bayesian solver across canonical damaged Linear A inscriptions."""
    console.print(Panel("[bold cyan]Multilateral Joint Bayesian Lacunae Solver[/bold cyan]\n[dim]Reconstructing effaced signs via arithmetic, liturgy, prosopography, and toponyms[/dim]"))

    from linear_a.predictive.multilateral_solver import MultilateralLacunaeSolver
    solver = MultilateralLacunaeSolver()
    report = solver.solve_all()

    results = report.solved_results
    if genre:
        results = [r for r in results if r.entry.genre.lower() == genre.lower()]

    table = Table(title=f"Multilateral Lacunae Restoration Matrix ({len(results)} Inscriptions)", show_header=True)
    table.add_column("Doc / Site", style="bold cyan", width=14)
    table.add_column("Masked Ductus", style="yellow", width=18)
    table.add_column("Predicted Sign", justify="center", style="bold green", width=14)
    table.add_column("Completed Word", style="bold white", width=18)
    table.add_column("Grade & Tier", justify="center", style="magenta", width=18)
    table.add_column("Bayes Factor", justify="right", style="cyan", width=12)
    table.add_column("Verification Method & Rationale", style="dim")

    for r in results:
        table.add_row(
            f"{r.entry.document}\n({r.entry.site})",
            r.entry.masked_token,
            f"[bold green]{r.predicted_sign}[/bold green]",
            r.entry.completed_word,
            f"{r.epistemic_grade}",
            f"{r.bayes_factor:.1f}",
            r.synthesis_notes,
        )

    console.print(table)
    console.print(f"\n[bold green]Summary:[/bold green] {report.summary}")


@app.command()
def syntax(
    target_id: str = typer.Argument(..., help="Curated tablet or libation vessel ID, e.g. HT_009 or IO_Za_002"),
):
    """Show a rule-based structural hypothesis for a curated reading."""
    from linear_a.grammar.pcfg_engine import StructuralPatternParser
    from linear_a.grammar.tree_renderer import render_ascii_tree

    reader = InterlinearReader()
    tablet = get_tablet_by_id(target_id)
    if tablet:
        document = reader.parse_tablet(tablet)
    else:
        vessel = next((item for item in LibationEngine().vessels if item.id == target_id), None)
        document = reader.parse_vessel(vessel) if vessel else None
    if document is None:
        console.print(f"[bold red]Error:[/bold red] Curated inscription '{target_id}' was not found.")
        raise typer.Exit(1)

    words = [token.transliteration for line in document.lines for token in line.tokens]
    result = StructuralPatternParser().parse_inscription(document.id, words)
    console.print(Panel(
        f"[bold cyan]Structural Pattern Hypothesis: {document.id}[/bold cyan]\n"
        f"[dim]Pattern coverage: {result.pattern_coverage:.0%} | This output does not establish grammar, language, meaning, or decipherment.[/dim]"
    ))
    table = Table(title="Observed Tokens and Pattern Labels", show_header=True)
    table.add_column("Observed token", style="cyan")
    table.add_column("Pattern label", style="yellow")
    for word, label in result.terminals:
        table.add_row(word, label)
    console.print(table)
    if result.parse_tree:
        console.print("[bold]Structural pattern tree[/bold]")
        console.print(render_ascii_tree(result.parse_tree))
    console.print(f"[dim]{result.limitations}[/dim]")


@app.command(name="evaluate-restorations")
def evaluate_restorations(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path for the evaluation report"),
    review_handoff: Optional[str] = typer.Option(None, "--review-handoff", help="Optional JSON path for the external review queue"),
    adjudications: Optional[str] = typer.Option(None, "--adjudications", "-a", help="Optional JSON/YAML path containing external reviewer adjudications"),
    include_exploratory: bool = typer.Option(False, "--include-exploratory", help="Also print project-curated exploratory metrics"),
):
    """Evaluate only externally accepted source-linked restoration references."""
    from linear_a.predictive.restoration_evaluation import RestorationEvaluationEngine

    engine = RestorationEvaluationEngine()
    bench = engine.source_linked_benchmark.load_external_adjudications(adjudications) if adjudications else None
    report = engine.evaluate(benchmark=bench)
    console.print(Panel(
        "[bold cyan]Source-Linked Restoration Benchmark[/bold cyan]\n"
        f"[dim]Benchmark: {report.benchmark_status} | Accepted scoreable references: {report.status_counts['accepted']}/{report.total_entries}[/dim]"
    ))
    console.print(
        f"[dim]Unadjudicated: {report.status_counts['unadjudicated']} | Disputed: {report.status_counts['disputed']} | Rejected: {report.status_counts['rejected']}[/dim]"
    )
    table = Table(title="Held-Out Evaluation Metrics", show_header=True)
    table.add_column("Method", style="cyan")
    table.add_column("References", justify="right")
    table.add_column("Attempted", justify="right")
    table.add_column("Abstained", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Top-1 precision", justify="right")
    table.add_column("Top-3 accuracy", justify="right")
    for metric in (report.template, report.phonotactic):
        table.add_row(
            metric.method,
            str(metric.references),
            str(metric.attempted),
            str(metric.abstained),
            f"{metric.coverage_pct:.1f}%",
            f"{metric.precision_top1_pct:.1f}%",
            f"{metric.top3_accuracy_pct:.1f}%",
        )
    console.print(table)
    console.print(f"[dim]{report.limitations}[/dim]")
    if include_exploratory:
        exploratory = engine.evaluate_exploratory()
        console.print("[yellow]Project-curated exploratory metrics (not independently adjudicated)[/yellow]")
        exploratory_table = Table(show_header=True)
        exploratory_table.add_column("Method", style="yellow")
        exploratory_table.add_column("References", justify="right")
        exploratory_table.add_column("Top-1 precision", justify="right")
        for metric in (exploratory.template, exploratory.phonotactic, exploratory.arithmetic_controls):
            exploratory_table.add_row(metric.method, str(metric.references), f"{metric.precision_top1_pct:.1f}%")
        console.print(exploratory_table)
    if export:
        output = Path(export)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Evaluation report written to {output}[/green]")
    if review_handoff:
        output = Path(review_handoff)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(engine.source_linked_benchmark.review_handoff(), indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Review handoff written to {output}[/green]")


@app.command()
def census_lacunae(
    site: Optional[str] = typer.Option(None, "--site", "-s", help="Filter by site (e.g. 'Haghia Triada', 'Khania')"),
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Filter by evidence status: E1, E3, E4, E2 (open), E0"),
    limit: int = typer.Option(25, "--limit", "-n", help="Maximum entries to display in detailed table"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional YAML path to export census"),
):
    """Catalog damaged tokens and evidence statuses in the local corpus snapshot."""
    console.print(Panel("[bold cyan]Corpus-Wide Linear A Epigraphic Lacunae Census[/bold cyan]\n[dim]Scanning all 1,720+ inscriptions across 20+ Minoan sites (SigLA / GORILA)[/dim]"))

    from linear_a.predictive.corpus_census import CorpusLacunaeCensusEngine
    engine = CorpusLacunaeCensusEngine()
    try:
        report = engine.run_census()
    except RuntimeError as exc:
        console.print(f"[bold red]Census source error:[/bold red] {exc}")
        raise typer.Exit(1)

    # Overview Metrics Table
    summary_table = Table(title="Corpus-Wide Damaged-Token Evidence Census", show_header=True)
    summary_table.add_column("Metric", style="bold white", width=36)
    summary_table.add_column("Count", justify="right", style="bold cyan", width=14)
    summary_table.add_column("Percentage / Epistemic Grounding", style="dim")

    summary_table.add_row("Total Inscriptions Scanned", f"{report.total_inscriptions_scanned:,}", "Verified local source snapshot")
    summary_table.add_row("Total Words / Sign Groups Scanned", f"{report.total_tokens_scanned:,}", "Observed textual tokens")
    summary_table.add_row("Total Damaged / Effaced Sign Instances", f"{report.total_damaged_tokens:,}", "Cataloged Lacuna Tokens")
    summary_table.add_row("Accounting-context entries (E1)", f"[yellow]{report.accounting_context_count:,}[/yellow]", "Observed numbers; no unique residual solution inferred")
    summary_table.add_row("Verified arithmetic candidates (E3)", f"[bold green]{report.deterministic_e3_count:,}[/bold green]", "Requires independent exact ledger-residual verification")
    summary_table.add_row("Formula-pattern candidates (E4)", f"[bold cyan]{report.sacred_liturgy_e4_count:,}[/bold cyan]", "Recurring curated formula pattern")
    summary_table.add_row("Attested-template candidates (E4)", f"[bold cyan]{report.attested_template_e4_count:,}[/bold cyan]", "Template match; unverified hypothesis")
    summary_table.add_row("Open phonotactic contexts (E2)", f"[yellow]{report.phonotactic_e2_count:,}[/yellow]", "Descriptive context; no sign proposal or score")
    summary_table.add_row("Unconstrained damaged entries (E0)", f"[red]{report.irrecoverable_e0_count:,}[/red]", "No candidate suggested")
    summary_table.add_row("CANDIDATE-SUPPORTED ENTRIES", f"[bold green]{report.total_recoverable_count:,}[/bold green]", f"[bold green]{report.information_recoverability_pct:.1f}% of damaged entries; excludes open E2 contexts[/bold green]")

    console.print(summary_table)

    # Site Breakdown Table
    site_table = Table(title="Regional Site Breakdown (Top 8 Centers)", show_header=True)
    site_table.add_column("Archaeological Site", style="bold cyan", width=22)
    site_table.add_column("Total Tokens", justify="right", width=14)
    site_table.add_column("Damaged", justify="right", style="yellow", width=12)
    site_table.add_column("Candidate-supported", justify="right", style="bold green", width=20)
    site_table.add_column("Candidate rate", justify="right", style="bold green", width=20)

    sorted_sites = sorted(report.site_breakdown.items(), key=lambda x: x[1]["damaged_tokens"], reverse=True)[:8]
    for s_name, s_data in sorted_sites:
        d = s_data["damaged_tokens"]
        r = s_data["recoverable"]
        pct = (r / d * 100.0) if d else 0.0
        site_table.add_row(
            s_name,
            f"{s_data['total_tokens']:,}",
            f"{d:,}",
            f"{r:,}",
            f"{pct:.1f}%",
        )

    console.print("\n", site_table)

    # Detailed Entries Filter
    entries = report.census_entries
    if site:
        entries = [e for e in entries if site.lower() in e.site.lower()]
    if tier:
        entries = [e for e in entries if tier.upper() in e.recoverability_tier]

    entries_to_show = entries[:limit]
    detail_table = Table(title=f"Sample Census Entries ({len(entries_to_show)} of {len(entries)} matching)", show_header=True)
    detail_table.add_column("Doc / Site", style="bold cyan", width=14)
    detail_table.add_column("Damaged Token", style="yellow", width=16)
    detail_table.add_column("Evidence status", justify="center", style="magenta", width=22)
    detail_table.add_column("Hypothesis", style="bold green", width=20)
    detail_table.add_column("Weight", justify="right", style="cyan", width=10)
    detail_table.add_column("Epigraphic Rationale", style="dim")

    for e in entries_to_show:
        target = f"{e.suggested_infill} -> {e.completed_word}" if e.completed_word else (e.suggested_infill or "-")
        bf_str = f"{e.bayes_factor:.1f}" if e.bayes_factor < 1000 else f"{e.bayes_factor:,.0f}"
        detail_table.add_row(
            f"{e.document}\n({e.site})",
            e.transliteration,
            e.recoverability_tier,
            target,
            bf_str,
            e.epigraphic_rationale,
        )

    console.print("\n", detail_table)
    console.print("[dim]Open E2 contexts carry no sign proposal or score. Other suggested infills are unverified structural hypotheses, not recovered text or decipherment results.[/dim]")

    if export:
        out_p = Path(export)
        engine.save_census_yaml(report, out_p)
        console.print(f"\n[bold green]✓ Census YAML catalog successfully exported to:[/bold green] {out_p}")


@app.command(name="segment-morphology")
def segment_morphology(
    votive_only: bool = typer.Option(False, "--votive-only", help="Focus solely on peak sanctuary libation vessels"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export morphological report"),
):
    """Unsupervised Minimum Description Length (MDL) morphological grammar induction."""
    from linear_a.morphology.bayesian_segmenter import BayesianMorphologicalSegmenter
    console.print(Panel("[bold cyan]Linear A Unsupervised Morphological Sieve (MDL Induction)[/bold cyan]\n[dim]Extracting prefix/suffix paradigms and alternating stem triplets[/dim]"))

    segmenter = BayesianMorphologicalSegmenter()
    report = segmenter.run_induction()

    console.print(f"[bold]Total Tokens Evaluated:[/bold] {report.total_tokens_evaluated:,} ({report.unique_types:,} unique types)")
    console.print(f"[bold]MDL Compression Ratio:[/bold] {report.compression_ratio:.2f}x (Bit savings: {report.mdl_raw_bits - report.mdl_compressed_bits:,.1f} bits)\n")

    # Table 1: Top Productive Morphemes
    affix_table = Table(title="Top Induced Morphemes (Prefixes & Suffixes)", show_header=True)
    affix_table.add_column("Type", style="bold magenta", width=12)
    affix_table.add_column("Form", style="bold cyan", width=10)
    affix_table.add_column("Stem Count", justify="right", width=12)
    affix_table.add_column("Frequency", justify="right", width=12)
    affix_table.add_column("Attested Stem Examples", style="dim")

    for p in report.top_prefixes[:6]:
        affix_table.add_row("PREFIX", f"{p.form}-", str(p.stem_count), str(p.frequency), ", ".join(p.stems[:4]))
    for s in report.top_suffixes[:6]:
        affix_table.add_row("SUFFIX", f"-{s.form}", str(s.stem_count), str(s.frequency), ", ".join(s.stems[:4]))

    console.print(affix_table, "\n")

    # Table 2: Alternating Stem Triplets (Kober Patterns)
    alt_table = Table(title="Discovered Stem Alternation Triplets (Kober Triplets)", show_header=True)
    alt_table.add_column("Stem Root", style="bold green", width=16)
    alt_table.add_column("Domain", width=14)
    alt_table.add_column("Attested Invariant Frames", style="cyan")

    filtered_alts = [a for a in report.stem_alternations if not votive_only or a.is_votive]
    for a in filtered_alts[:10]:
        dom = "[bold yellow]VOTIVE[/bold yellow]" if a.is_votive else "[dim]ADMIN[/dim]"
        frames = " · ".join(f"{v['prefix']}+{a.stem}+{v['suffix']}" for v in a.variants[:4])
        alt_table.add_row(a.stem, dom, frames)

    console.print(alt_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Morphological report exported to {out_p}[/green]")


@app.command(name="solve-diophantine")
def solve_diophantine(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export Diophantine solutions"),
):
    """Solve rational equations across balanced Minoan tablets with exact zero residual."""
    from linear_a.accounting.diophantine_solver import DiophantineTabletSolver
    console.print(Panel("[bold cyan]Linear A Diophantine Multi-Fraction Solver[/bold cyan]\n[dim]Verifying exact rational conservation and synthetic mask recoveries[/dim]"))

    solver = DiophantineTabletSolver()
    result = solver.benchmark_synthetic_masks()

    console.print(f"[bold]Balanced Tablets Tested:[/bold] {result.tablets_tested}")
    console.print(f"[bold]Total Masks Tested:[/bold] {result.total_masks}")
    console.print(f"[bold]Exact Diophantine Recoveries:[/bold] [bold green]{result.exact_recoveries} / {result.total_masks} ({result.accuracy_pct:.1f}%)[/bold green]\n")

    sol_table = Table(title=f"Sample Diophantine Solution Certificates ({min(8, len(result.solutions))} items)", show_header=True)
    sol_table.add_column("Tablet", style="bold cyan", width=10)
    sol_table.add_column("Variable", style="yellow", width=18)
    sol_table.add_column("Known Sum", justify="right", width=12)
    sol_table.add_column("Total", justify="right", width=12)
    sol_table.add_column("Residual", justify="right", style="bold green", width=12)
    sol_table.add_column("Minoan Solution", style="bold magenta", width=16)

    for sol in result.solutions[:8]:
        sym_str = f" + {sol.fraction_symbols}" if sol.fraction_symbols else ""
        sol_table.add_row(
            sol.tablet_id,
            sol.target_variable,
            str(sol.known_sum),
            str(sol.stated_total),
            f"{float(sol.residual):.3f}",
            f"{sol.integer_part}{sym_str}",
        )

    console.print(sol_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tablets_tested": result.tablets_tested,
            "total_masks": result.total_masks,
            "accuracy_pct": result.accuracy_pct,
            "solutions": [s.to_dict() for s in result.solutions],
        }
        out_p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Diophantine solutions exported to {out_p}[/green]")


@app.command(name="typological-profile")
def typological_profile(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export typological profile"),
):
    """Compute information entropy and typological distance against Bronze Age language families."""
    from linear_a.skeptic.typological_profiler import TypologicalProfiler
    console.print(Panel("[bold cyan]Linear A Typological Linguistic Profiler[/bold cyan]\n[dim]Information entropy, syllable canonical shapes, and structural language distance[/dim]"))

    profiler = TypologicalProfiler()
    report = profiler.analyze_profile()

    console.print(f"[bold]Total Words Analyzed:[/bold] {report.total_words_analyzed:,}")
    console.print(f"[bold]Mean Word Length:[/bold] {report.mean_morae_per_word:.2f} morae (Votive: {report.votive_mean_morae:.2f} | Admin: {report.admin_mean_morae:.2f})")
    console.print(f"[bold]Open-Syllable Ratio (CV):[/bold] {report.open_syllable_ratio * 100:.1f}%")
    console.print(f"[bold]Unigram Sign Entropy (H₁):[/bold] {report.unigram_entropy_bits:.2f} bits | [bold]Bigram Conditional Entropy (H₂):[/bold] {report.bigram_entropy_bits:.2f} bits")
    console.print(f"[bold]Agglutination Index:[/bold] {report.agglutination_index:.2f}\n")

    rank_table = Table(title="Cross-Linguistic Structural Distance Rankings", show_header=True)
    rank_table.add_column("Language Family", style="bold cyan", width=28)
    rank_table.add_column("Typological Class", width=24)
    rank_table.add_column("Distance", justify="right", width=12)
    rank_table.add_column("Compatibility", justify="right", style="bold green", width=14)
    rank_table.add_column("Structural Assessment", style="dim")

    for m in report.distance_rankings:
        color = "bold green" if m.compatibility_score >= 70.0 else ("yellow" if m.compatibility_score >= 50.0 else "red")
        rank_table.add_row(
            m.benchmark_name,
            m.family_group,
            f"{m.structural_distance:.3f}",
            f"[{color}]{m.compatibility_score:.1f}%[/{color}]",
            m.verdict,
        )

    console.print(rank_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Typological profile exported to {out_p}[/green]")


@app.command(name="votive-grammar")
def votive_grammar(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export votive grammar report"),
):
    """Model clausal syntax and Markov phase transitions across peak sanctuary vessels."""
    from linear_a.votive.clausal_grammar import VotiveClausalGrammarEngine
    console.print(Panel("[bold cyan]Linear A Votive Clausal Grammar & Regional Liturgy Engine[/bold cyan]\n[dim]Markov phase transitions and liturgical variation across peak sanctuaries[/dim]"))

    engine = VotiveClausalGrammarEngine()
    report = engine.evaluate_votive_grammar()

    console.print(f"[bold]Total Inscriptions Parsed:[/bold] {report.total_vessels_parsed}")
    console.print(f"[bold]Canonical Syntax Conformance:[/bold] [bold green]{report.canonical_syntax_conformance_pct:.1f}%[/bold green]\n")

    # Table 1: Regional Profiles
    reg_table = Table(title="Regional Liturgical Profiles & Epiclesis Variants", show_header=True)
    reg_table.add_column("Region", style="bold cyan", width=32)
    reg_table.add_column("Sites", width=18)
    reg_table.add_column("Vessels", justify="right", width=10)
    reg_table.add_column("Header", style="yellow", width=18)
    reg_table.add_column("Epithet / Theonym", style="bold magenta", width=20)
    reg_table.add_column("Regional Characteristics", style="dim")

    for r in report.regional_profiles:
        reg_table.add_row(
            r.region_name,
            ", ".join(r.sites),
            str(r.vessel_count),
            r.header_variant,
            r.epithet_variant,
            r.notes,
        )

    console.print(reg_table, "\n")

    # Table 2: Markov Phase Transitions
    trans_table = Table(title="Liturgical Phase Transition Matrix (P(next | current))", show_header=True)
    trans_table.add_column("Current Liturgical State", style="bold magenta", width=28)
    trans_table.add_column("Most Probable Next State", style="bold green", width=28)
    trans_table.add_column("Transition Probability", justify="right", width=22)

    probs = report.markov_matrix.transition_probabilities
    for state, destinations in probs.items():
        if destinations and state != "END":
            best_dest, best_p = max(destinations.items(), key=lambda x: x[1])
            if best_p > 0:
                trans_table.add_row(
                    state.replace("PHASE_", "Phase ").replace("_", " "),
                    best_dest.replace("PHASE_", "Phase ").replace("_", " "),
                    f"{best_p * 100:.1f}%",
                )

    console.print(trans_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Votive grammar report exported to {out_p}[/green]")


@app.command(name="analyze-ligatures")
def analyze_ligatures(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export ligature taxonomy"),
):
    """Decompose composite monograms and analyze scriptorium commodity repertoires."""
    from linear_a.palaeography.ligature_taxonomy import LigatureTaxonomyEngine
    console.print(Panel("[bold cyan]Linear A Paleographic Ligature Taxonomy & Scriptorium Engine[/bold cyan]\n[dim]Decomposing composite monograms and identifying scribal fingerprints[/dim]"))

    engine = LigatureTaxonomyEngine()
    report = engine.generate_taxonomy_report()

    console.print(f"[bold]Total Cataloged Monograms:[/bold] {report.total_ligatures_cataloged}")
    console.print(f"[bold]Total Epigraphic Instances:[/bold] {report.total_instances_attested:,}")
    comms_str = ", ".join(f"{k}: {v}" for k, v in sorted(report.commodity_classes.items()))
    console.print(f"[bold]Commodity Base Breakdown:[/bold] {comms_str}\n")

    # Table 1: Decomposed Monograms & Linear B Parallels
    mono_table = Table(title="Composite Monograms & Adjunct Modifiers", show_header=True)
    mono_table.add_column("Monogram", style="bold cyan", width=12)
    mono_table.add_column("Base Ideogram", width=16)
    mono_table.add_column("Adjunct Modifier", width=16)
    mono_table.add_column("Functional Role", style="magenta", width=22)
    mono_table.add_column("Mycenaean Linear B Parallel", style="bold green", width=34)

    for d in report.decomposed_records[:10]:
        lb_str = d.linear_b_parallel or "[dim]Unattested in Linear B[/dim]"
        mono_table.add_row(
            d.notation,
            f"{d.base_commodity} ({d.base_name})",
            f"{d.modifier_sign} ({d.modifier_reading})",
            d.modifier_role,
            lb_str,
        )

    console.print(mono_table, "\n")

    # Table 2: Scriptorium Specialization
    script_table = Table(title="Palatial Scriptorium Commodity Repertoires", show_header=True)
    script_table.add_column("Scriptorium Archive", style="bold cyan", width=24)
    script_table.add_column("Ligatures", justify="right", width=12)
    script_table.add_column("Dominant Commodities", width=28)
    script_table.add_column("Scribal Specialization", style="dim")

    for s in report.scriptorium_profiles:
        comm_list = ", ".join(f"{c[0]} ({c[1]})" for c in s.dominant_commodities)
        script_table.add_row(
            s.site_name,
            str(s.total_ligatures_attested),
            comm_list,
            s.scribal_specialization,
        )

    console.print(script_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Ligature taxonomy exported to {out_p}[/green]")


@app.command(name="induce-substratum")
def induce_substratum(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export substratum report"),
):
    """Induce latent Minoan phonological features and evaluate Knossos pre-Greek substrate."""
    from linear_a.phonology.substratum_induction import MinoanSubstratumInducer
    console.print(Panel("[bold cyan]Minoan Phonological Substratum & Adaptation Matrix[/bold cyan]\n[dim]Tracing script adaptation rules and pre-Greek Knossos substrate toponyms[/dim]"))

    inducer = MinoanSubstratumInducer()
    report = inducer.generate_substratum_report()

    console.print(f"[bold]Total Substrate Lexicon Entries:[/bold] {report.total_substrate_entries}")
    console.print(f"[bold]Direct Homologies (Linear A = Linear B):[/bold] [bold green]{report.direct_homologies_count}[/bold green]")
    console.print(f"[bold]Phonetic Cognates (Shifted Vowels):[/bold] [bold cyan]{report.phonetic_cognates_count}[/bold cyan]")
    console.print(f"[bold]Substrate Retention Rate:[/bold] [bold green]{report.homology_rate_pct:.1f}%[/bold green]\n")

    # Table 1: Voicing Neutrality
    voice_table = Table(title="Minoan Consonant Voicing Neutrality Matrix", show_header=True)
    voice_table.add_column("Consonant Series", style="bold cyan", width=22)
    voice_table.add_column("Linear A Series", justify="right", width=16)
    voice_table.add_column("Linear B Series", justify="right", width=16)
    voice_table.add_column("Neutrality", justify="right", style="bold green", width=14)
    voice_table.add_column("Palaeographic Explanation", style="dim")

    for v in report.voicing_metrics:
        voice_table.add_row(
            v.series_name,
            str(v.linear_a_distinct_series),
            str(v.linear_b_distinct_series),
            f"{v.neutrality_ratio * 100.0:.0f}%",
            v.epigraphic_explanation,
        )
    console.print(voice_table, "\n")

    # Table 2: Toponym Homologies
    top_table = Table(title="Knossos Pre-Greek Substrate Toponyms (Sample)", show_header=True)
    top_table.add_column("Linear B", style="bold yellow", width=14)
    top_table.add_column("Linear A Cognate", style="bold cyan", width=16)
    top_table.add_column("Classical Name", width=24)
    top_table.add_column("Region", width=16)
    top_table.add_column("Preservation Status", style="bold green", width=20)
    top_table.add_column("Phonetic Shift Notes", style="dim")

    for e in report.entries[:10]:
        top_table.add_row(
            e.linear_b_form,
            e.linear_a_form or "—",
            e.classical_name,
            e.region,
            e.preservation_status,
            e.vowel_harmony_notes,
        )
    console.print(top_table)

    console.print(f"\n[bold magenta]Vowel-O Deficiency Assessment:[/bold magenta] {report.o_deficiency.verdict}")

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Substratum report exported to {out_p}[/green]")


@app.command(name="solve-multi-commodity")
def solve_multi_commodity(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export multi-commodity report"),
):
    """Solve simultaneous rational Diophantine equations across multi-commodity ledgers."""
    from linear_a.accounting.multi_commodity_solver import MultiCommodityDiophantineSolver
    console.print(Panel("[bold cyan]Multi-Commodity Metrological Diophantine Solver[/bold cyan]\n[dim]Simultaneous rational ledger recovery across dry and liquid volume hierarchies[/dim]"))

    solver = MultiCommodityDiophantineSolver()
    report = solver.benchmark_multi_commodity_solver()

    console.print(f"[bold]Multi-Commodity Ledgers Evaluated:[/bold] {report.total_multi_commodity_tablets} ({', '.join(report.tablets_evaluated)})")
    console.print(f"[bold]Simultaneous Mask Permutations:[/bold] {report.simultaneous_benchmark_masks}")
    console.print(f"[bold]Exact Rational Recoveries:[/bold] [bold green]{report.exact_recoveries_count}[/bold green]")
    console.print(f"[bold]Diophantine Recovery Accuracy:[/bold] [bold green]{report.accuracy_pct:.1f}%[/bold green]\n")

    # Table 1: Commodity-Fraction Associations
    assoc_table = Table(title="Corpus Commodity-Fraction Associations", show_header=True)
    assoc_table.add_column("Commodity", style="bold cyan", width=14)
    assoc_table.add_column("Metrology Class", width=18)
    assoc_table.add_column("Attestations", justify="right", width=14)
    assoc_table.add_column("Dominant Fraction", style="bold yellow", justify="center", width=18)
    assoc_table.add_column("Mean Subunit Remainder", justify="right", width=22)

    for ca in report.commodity_associations:
        assoc_table.add_row(
            ca.commodity,
            ca.metrology_class,
            str(ca.total_attestations),
            ca.dominant_fraction,
            f"{ca.mean_fractional_remainder:.3f}",
        )
    console.print(assoc_table, "\n")

    # Table 2: Solutions Sample
    sol_table = Table(title="Simultaneous Multi-Commodity Solutions Sample", show_header=True)
    sol_table.add_column("Tablet", style="bold cyan", width=10)
    sol_table.add_column("Masked Variables", width=36)
    sol_table.add_column("True Values", width=20)
    sol_table.add_column("Solved Minoan Symbols", style="bold green", width=22)
    sol_table.add_column("Proof Certificate", style="dim")

    for s in report.solutions[:8]:
        sol_table.add_row(
            s.tablet_id,
            ", ".join(s.masked_variables),
            ", ".join(str(v) for v in s.true_values),
            ", ".join(s.minoan_symbols),
            s.proof_certificate,
        )
    console.print(sol_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Multi-commodity report exported to {out_p}[/green]")


@app.command(name="cluster-ductus")
def cluster_ductus(
    k: int = typer.Option(4, "--k", "-k", help="Number of latent scribal clusters"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export ductus report"),
):
    """Cluster scribal hands by stroke ductus and evaluate LM IB cross-site mobility."""
    from linear_a.palaeography.ductus_clustering import DuctusClusteringEngine
    console.print(Panel("[bold cyan]Unsupervised Scribal Hand Ductus Clustering Engine[/bold cyan]\n[dim]Palaeographic vector clustering and LM IB destruction horizon mobility tracking[/dim]"))

    engine = DuctusClusteringEngine()
    report = engine.generate_ductus_report()

    console.print(f"[bold]Total Tablets Profiled:[/bold] {report.total_tablets_profiled}")
    console.print(f"[bold]Discovered Latent Hands:[/bold] {report.optimal_k_clusters}")
    console.print(f"[bold]Mean Silhouette Score:[/bold] [bold green]{report.mean_silhouette_score:.3f}[/bold green]\n")

    # Table 1: Discovered Hands
    hand_table = Table(title="Discovered Latent Scribal Hands", show_header=True)
    hand_table.add_column("Hand Name", style="bold cyan", width=34)
    hand_table.add_column("Primary Center", width=18)
    hand_table.add_column("Tablets", justify="right", width=10)
    hand_table.add_column("Scribal Specialization", style="bold yellow", width=32)
    hand_table.add_column("Homogeneity", justify="right", style="bold green", width=14)

    for h in report.hands_discovered:
        hand_table.add_row(
            h.hand_name,
            h.primary_site,
            str(h.total_tablets),
            h.scribal_specialization,
            f"{h.homogeneity_score * 100.0:.0f}%",
        )
    console.print(hand_table, "\n")

    # Table 2: Cross-site Mobility
    mob_table = Table(title="LM IB Cross-Site Scriptorium Mobility Matches", show_header=True)
    mob_table.add_column("Tablet", style="bold cyan", width=12)
    mob_table.add_column("Source Site", width=16)
    mob_table.add_column("Matched Central Hand", width=34)
    mob_table.add_column("Similarity", justify="right", style="bold green", width=12)
    mob_table.add_column("Itinerant Scribe?", justify="center", width=18)

    for m in report.cross_site_mobility_matches:
        mob_str = "[bold green]YES[/bold green]" if m.is_plausible_itinerant_scribe else "[dim]LOCAL SCRIPT[/dim]"
        mob_table.add_row(
            m.tablet_id,
            m.source_site,
            m.matched_hand_name,
            f"{m.palaeographic_similarity_pct:.1f}%",
            mob_str,
        )
    console.print(mob_table)
    console.print(f"\n[bold magenta]Historical Synthesis:[/bold magenta] {report.mobility_verdict}")

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Ductus report exported to {out_p}[/green]")


@app.command(name="review-adjudications")
def review_adjudications(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export portal report"),
):
    """Display epigrapher peer-review dossier packets and instructions for external validation."""
    from linear_a.predictive.adjudication_portal import AdjudicationPortalEngine
    console.print(Panel("[bold cyan]Epigrapher Peer-Review Adjudication Portal[/bold cyan]\n[dim]Formal dossier packets for candidate-supported damaged tokens (LADP v1.0 Section 18)[/dim]"))

    engine = AdjudicationPortalEngine()
    report = engine.generate_portal_report()

    console.print(f"[bold]Total Candidate Dossiers Prepared:[/bold] {report.total_candidate_packets}")
    console.print(f"[bold]Currently Adjudicated Tokens:[/bold] {report.adjudicated_count}")
    console.print(f"[bold]Adjudication Standards:[/bold] Requires primary publication citation (e.g. GORILA Vol. I-V)\n")

    pack_table = Table(title="Candidate-Supported Damaged Token Dossiers (Sample)", show_header=True)
    pack_table.add_column("Token ID", style="bold cyan", width=14)
    pack_table.add_column("Document", width=12)
    pack_table.add_column("Site / Carrier", width=22)
    pack_table.add_column("Damaged Glyph", style="bold yellow", width=16)
    pack_table.add_column("Proposed Sign", style="bold green", width=14)
    pack_table.add_column("Evidence Tier", justify="center", width=14)
    pack_table.add_column("Primary Citation Locator", style="dim")

    for p in report.review_packets[:8]:
        pack_table.add_row(
            p.token_id,
            p.document_id,
            f"{p.site} ({p.carrier[:12]})",
            p.surviving_glyph,
            f"{p.proposed_sign} ({p.completed_word})",
            p.evidence_tier,
            p.primary_edition_locator or "Unavailable",
        )
    console.print(pack_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Portal report exported to {out_p}[/green]")


@app.command(name="analyze-dialectology")
def analyze_dialectology(
    permutations: int = typer.Option(1000, "--permutations", "-p", help="Number of Monte Carlo permutations for Mantel test"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export dialectology report"),
):
    """Evaluate regional geographical dialect variation and execute a permutation Mantel test."""
    from linear_a.dialect.geographical_dialectology import GeographicalDialectologyEngine
    console.print(Panel("[bold cyan]Geographical Dialectology & Spatial Distance Matrix[/bold cyan]\n[dim]Great-Circle geodesic distance vs Jaccard dissimilarity (LADP v1.0 Section 19)[/dim]"))

    engine = GeographicalDialectologyEngine()
    report = engine.generate_dialectology_report(permutations=permutations)

    console.print(f"[bold]Total Provenances Profiled:[/bold] {report.total_sites_profiled}")
    console.print(f"[bold]Total Documents Analyzed:[/bold] {report.total_documents_analyzed}")
    console.print(f"[bold]Mean Geodesic Distance:[/bold] {report.mean_geographic_distance_km:.1f} km")
    console.print(f"[bold]Mean Jaccard Dissimilarity:[/bold] {report.mean_jaccard_dissimilarity:.3f}\n")

    # Table 1: Site Profiles
    site_table = Table(title="Regional Archaeological Provenances", show_header=True)
    site_table.add_column("Site Code", style="bold cyan", width=12)
    site_table.add_column("Name & Region", width=36)
    site_table.add_column("Documents", justify="right", width=12)
    site_table.add_column("Vocab Size", justify="right", style="bold green", width=12)
    site_table.add_column("Top Commodities / Focus", style="bold yellow")

    for s in report.site_profiles:
        site_table.add_row(
            s.site_code,
            s.name,
            str(s.total_documents),
            str(s.vocabulary_size),
            ", ".join(s.dominant_commodities),
        )
    console.print(site_table, "\n")

    # Table 2: Mantel Test Result
    m = report.mantel_test
    console.print(Panel(
        f"[bold]Mantel Matrix Correlation (r_M):[/bold] [bold cyan]{m.correlation_r:.3f}[/bold cyan]\n"
        f"[bold]Empirical p-value ({m.permutations_count} perms):[/bold] [bold {'green' if m.p_value > 0.05 else 'magenta'}]{m.p_value:.4f}[/bold {'green' if m.p_value > 0.05 else 'magenta'}]\n"
        f"[bold]Null Permutation Distribution:[/bold] mean={m.null_mean_r:.3f}, std={m.null_std_r:.3f}\n"
        f"[bold]Epistemic Verdict:[/bold] [bold yellow]{m.epistemic_verdict}[/bold yellow]",
        title="Mantel Permutation Test Synthesis",
    ))

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Dialectology report exported to {out_p}[/green]")


@app.command(name="analyze-script-phylogeny")
def analyze_script_phylogeny(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export phylogeny report"),
):
    """Model the phylogenetic lineage across Cretan Hieroglyphic, Linear A, Linear B, and Cypro-Minoan."""
    from linear_a.phylogeny.script_lineage import AegeanScriptPhylogenyEngine
    console.print(Panel("[bold cyan]Aegean Bronze Age Script Phylogeny & Lineage Engine[/bold cyan]\n[dim]Graphemic drift, stroke reduction, and entropy transmission (LADP v1.0 Section 20)[/dim]"))

    engine = AegeanScriptPhylogenyEngine()
    report = engine.generate_phylogeny_report()

    console.print(f"[bold]Cataloged Sign Homologues:[/bold] {report.total_homologues_cataloged}")
    console.print(f"[bold]Cretan Hieroglyphic -> Linear A Retention:[/bold] [bold green]{report.chic_to_linear_a_retention_pct:.1f}%[/bold green]")
    console.print(f"[bold]Linear A -> Linear B Retention:[/bold] [bold green]{report.linear_a_to_linear_b_retention_pct:.1f}%[/bold green]")
    console.print(f"[bold]Linear A -> Cypro-Minoan Retention:[/bold] [bold yellow]{report.linear_a_to_cypro_minoan_retention_pct:.1f}%[/bold yellow]")
    console.print(f"[bold]Stroke Simplification (Hieroglyphic -> Linear A):[/bold] [bold cyan]{report.mean_stroke_reduction_chic_to_la_pct:.1f}% reduction[/bold cyan]\n")

    # Table: Script Lineage Nodes
    node_table = Table(title="Aegean Writing Systems Lineage", show_header=True)
    node_table.add_column("Script", style="bold cyan", width=16)
    node_table.add_column("Chronology", width=22)
    node_table.add_column("Approx BCE", width=16)
    node_table.add_column("Parent Script", width=22)
    node_table.add_column("Entropy (bits)", justify="right", style="bold green", width=14)
    node_table.add_column("Mean Strokes", justify="right", style="bold yellow", width=14)

    for sc in report.scripts_profiled:
        node_table.add_row(
            sc.script_id,
            sc.chronological_range,
            sc.approx_bce,
            sc.parent_script or "Genesis Root",
            f"{sc.shannon_entropy_bits:.2f}",
            f"{sc.mean_stroke_complexity:.1f}",
        )
    console.print(node_table, "\n")

    # Homologue Sample
    hom_table = Table(title="Aegean Sign Homologues (Sample)", show_header=True)
    hom_table.add_column("Canonical Name", style="bold cyan", width=20)
    hom_table.add_column("CHIC", width=12)
    hom_table.add_column("Linear A", style="bold green", width=12)
    hom_table.add_column("Linear B", width=12)
    hom_table.add_column("CM", width=10)
    hom_table.add_column("Reading", style="bold yellow", width=10)
    hom_table.add_column("Pictorial Origin", style="dim")

    for h in report.homologues[:8]:
        hom_table.add_row(
            h.canonical_name,
            h.chic_id or "—",
            h.linear_a_id,
            h.linear_b_id or "—",
            h.cypro_minoan_id or "—",
            h.phonetic_reading,
            h.pictorial_origin,
        )
    console.print(hom_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Phylogeny report exported to {out_p}[/green]")


@app.command(name="solve-unified-metrology")
def solve_unified_metrology(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export metrology report"),
):
    """Synthesize archaeological balance weights, fractional volume units, and commodity exchange ratios."""
    from linear_a.accounting.unified_metrology import MinoanUnifiedMetrologyEngine
    console.print(Panel("[bold cyan]Minoan Unified Metrological Weight & Commodity Tree[/bold cyan]\n[dim]Petruso balance weights & Ferrara fractional volume synthesis (LADP v1.0 Section 21)[/dim]"))

    engine = MinoanUnifiedMetrologyEngine()
    report = engine.generate_metrology_report()

    console.print(f"[bold]Base Minoan Weight Module (M):[/bold] [bold green]{report.base_weight_unit_grams} g[/bold green]")
    console.print(f"[bold]Base Volume Unit (Major Capacity):[/bold] [bold green]{report.base_volume_unit_liters} L[/bold green]")
    console.print(f"[bold]Minoan Talent Standard (L):[/bold] [bold cyan]{report.talent_subdivisions['talent_kg']} kg ({report.talent_subdivisions['light_minas_M']:.0f} M)[/bold cyan]\n")

    # Table 1: Balance Weights
    weight_table = Table(title="Archaeological Balance Weight Standards (Mochlos, Akrotiri, HT)", show_header=True)
    weight_table.add_column("Unit Symbol", style="bold cyan", width=14)
    weight_table.add_column("Unit Name", width=30)
    weight_table.add_column("Mass (grams)", justify="right", style="bold green", width=16)
    weight_table.add_column("Ratio to M", justify="right", width=14)
    weight_table.add_column("Key Attestation Site", style="dim")

    for bw in report.balance_weights:
        weight_table.add_row(
            bw.unit_symbol,
            bw.name,
            f"{bw.mass_grams:.2f} g",
            f"{bw.base_unit_ratio:.3f} M",
            bw.archaeological_attestation,
        )
    console.print(weight_table, "\n")

    # Table 2: Commodity Equivalences
    eq_table = Table(title="Palatial Commodity Equivalence Coefficients", show_header=True)
    eq_table.add_column("Commodity A", style="bold cyan", width=24)
    eq_table.add_column("Commodity B", width=22)
    eq_table.add_column("Exchange Ratio", justify="right", style="bold yellow", width=16)
    eq_table.add_column("Evidence Tier", justify="center", width=18)
    eq_table.add_column("Tablets Attested", style="bold green")

    for er in report.equivalence_ratios:
        eq_table.add_row(
            er.commodity_a,
            er.commodity_b,
            f"{er.canonical_ratio:.1f} : 1",
            er.evidence_tier,
            ", ".join(er.tablets_attested),
        )
    console.print(eq_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Metrology report exported to {out_p}[/green]")


@app.command(name="render-glyph-vectors")
def render_glyph_vectors(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export stroke vector report"),
):
    """Render procedural vector stroke primitives and contrast clay vs lapidary vs metalwork ductus."""
    from linear_a.palaeography.stroke_engine import PalaeographicStrokeEngine
    console.print(Panel("[bold cyan]Palaeographic Stroke Vector Engine & Dynamic Glyph Viewer[/bold cyan]\n[dim]Procedural SVG Bezier stroke primitives and carrier ductus differentiation (LADP v1.0 Section 22)[/dim]"))

    engine = PalaeographicStrokeEngine()
    report = engine.generate_stroke_report()

    console.print(f"[bold]Total Signs Vectorized:[/bold] {report.total_glyphs_vectorized}")
    console.print(f"[bold]Carriers Profiled:[/bold] {', '.join(report.carriers_profiled)}")
    console.print(f"[bold]Mean Stroke Count:[/bold] {report.mean_stroke_count:.1f}")
    console.print(f"[bold]Lapidary-to-Clay Angularity Ratio:[/bold] [bold cyan]{report.mean_lapidary_angularity / report.mean_clay_angularity:.2f}x[/bold cyan]\n")

    glyph_table = Table(title="Vectorized Linear A Syllabic Signs", show_header=True)
    glyph_table.add_column("Sign ID", style="bold cyan", width=12)
    glyph_table.add_column("Phonetic", style="bold green", width=12)
    glyph_table.add_column("Canonical Name", width=24)
    glyph_table.add_column("Strokes", justify="right", width=10)
    glyph_table.add_column("Angularity Ratio", justify="right", style="bold yellow", width=18)
    glyph_table.add_column("Epigraphic Facsimile Notes", style="dim")

    for g in report.glyphs:
        glyph_table.add_row(
            g.sign_id,
            g.phonetic_reading,
            g.canonical_name,
            str(g.stroke_count),
            f"{g.angularity_lapidary_vs_clay_ratio:.2f}x",
            g.epigraphic_notes,
        )
    console.print(glyph_table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Stroke vector report exported to {out_p}[/green]")


@app.command(name="phonetics-atlas")
def phonetics_atlas(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max number of signs to display"),
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Filter by confidence tier (E4, E3, E2, E0)"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export phonetics catalog"),
):
    """Inspect the 87-sign Ventris-Grid phonetic transfer atlas with acoustic formants and confidence tiers."""
    from linear_a.phonology.acoustic_reconstruction import AcousticReconstructionEngine
    console.print(Panel("[bold cyan]Aegean Syllabary Phonetics & Acoustic Formant Atlas[/bold cyan]\n[dim]Minoan 3-4 vowel space and Ventris-Grid transfer confidence tiers (LADP v1.0 Section 24)[/dim]"))

    engine = AcousticReconstructionEngine()
    profiles = engine.get_all_profiles()

    if tier:
        profiles = [p for p in profiles if p.confidence_tier.upper() == tier.upper()]
    if limit:
        profiles = profiles[:limit]

    table = Table(title="Linear A Syllabary Phonetic & Formant Catalog", show_header=True)
    table.add_column("Code", style="bold cyan", width=8)
    table.add_column("Sign", style="bold yellow", width=8)
    table.add_column("Linear B", width=10)
    table.add_column("IPA", style="bold green", width=14)
    table.add_column("Tier", justify="center", width=8)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("F1 (Hz)", justify="right", width=9)
    table.add_column("F2 (Hz)", justify="right", width=9)
    table.add_column("Manner", width=12)
    table.add_column("Notes", style="dim")

    for p in profiles:
        tier_color = "green" if p.confidence_tier == "E4" else ("yellow" if p.confidence_tier == "E3" else ("dark_orange" if p.confidence_tier == "E2" else "red"))
        table.add_row(
            p.sign_code,
            p.transliteration,
            p.linear_b_value,
            p.ipa_realization,
            f"[{tier_color}]{p.confidence_tier}[/{tier_color}]",
            f"{p.confidence_score * 100:.0f}%",
            f"{p.formant_f1:.0f}",
            f"{p.formant_f2:.0f}",
            p.consonant_manner,
            p.notes,
        )
    console.print(table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in profiles]
        out_p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Phonetics catalog exported to {out_p}[/green]")


@app.command(name="analyze-prosody")
def analyze_prosody(
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export prosody report"),
):
    """Analyze moraic scansion, rhythmic cadence, and poetic meter across peak sanctuary libation vessels."""
    from linear_a.phonology.prosodic_meter import ProsodicMeterEngine
    console.print(Panel("[bold cyan]Minoan Moraic Prosodic Scansion & Sacred Metric Cadence[/bold cyan]\n[dim]Metrical analysis of 26 peak sanctuary libation vessels (LADP v1.0 Section 23)[/dim]"))

    engine = ProsodicMeterEngine()
    analysis = engine.analyze_votive_corpus()

    console.print(f"[bold]Vessels Scanned:[/bold] {analysis['vessel_count']}")
    console.print(f"[bold]Mean Morae per Inscription:[/bold] {analysis['mean_morae_per_vessel']:.2f}")
    console.print(f"[bold]Dominant Sacred Meter:[/bold] [bold green]{analysis['dominant_corpus_meter']}[/bold green]")
    console.print(f"[bold]Meter Distribution:[/bold] {json.dumps(analysis['meter_distribution'])}\n")

    table = Table(title="Sanctuary Vessel Metric Scansions", show_header=True)
    table.add_column("Vessel ID", style="bold cyan", width=12)
    table.add_column("Site", style="yellow", width=20)
    table.add_column("Carrier", width=22)
    table.add_column("Morae", justify="right", width=8)
    table.add_column("Foot", style="bold green", width=14)
    table.add_column("Regularity", justify="right", width=12)
    table.add_column("Moraic Scansion Track", style="bold magenta")

    for r in analysis["vessel_reports"]:
        table.add_row(
            r["id"],
            r["site"],
            r["carrier"],
            str(r["total_morae"]),
            r["dominant_foot"],
            f"{r['regularity_score'] * 100:.0f}%",
            r["scansion_line"],
        )
    console.print(table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Prosodic analysis exported to {out_p}[/green]")


@app.command(name="recite-text")
def recite_text(
    inscription_id: str = typer.Argument("IO_Za_2", help="Target tablet or vessel ID (e.g. IO_Za_2, PK_Za_11, HT_085, PH_001)"),
    tempo: float = typer.Option(1.0, "--tempo", "-t", help="Playback tempo scaling factor (0.5 to 2.0)"),
    pitch: float = typer.Option(130.0, "--pitch", "-p", help="Base F0 pitch frequency in Hz (100.0 to 200.0)"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional JSON path to export recitation package"),
):
    """Generate multimodal acoustic recitation and Web Audio synthesis schedule for an inscription."""
    from linear_a.reading.reciter_engine import ReciterEngine
    console.print(Panel(f"[bold cyan]Minoan Epigraphic Reciter: {inscription_id}[/bold cyan]\n[dim]Acoustic formant reconstruction and Web Audio synthesis schedule (LADP v1.0 Section 23)[/dim]"))

    engine = ReciterEngine()
    pkg = engine.get_recitation_by_id(inscription_id)

    if not pkg:
        console.print(f"[bold red]Inscription '{inscription_id}' not found in curated flagship texts, libation vessels, or tablets.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold]Inscription ID:[/bold] {pkg.id} ({pkg.genre})")
    console.print(f"[bold]Site & Carrier:[/bold] {pkg.site} | {pkg.carrier}")
    console.print(f"[bold]Epigraphic Text:[/bold] [bold yellow]{pkg.raw_text}[/bold yellow]")
    console.print(f"[bold]Phonetic IPA:[/bold] [bold green]{pkg.ipa_text}[/bold green]")
    console.print(f"[bold]Moraic Scansion:[/bold] [bold magenta]{pkg.scansion_str}[/bold magenta] ({pkg.total_morae} morae, foot: {pkg.dominant_foot})")
    console.print(f"[bold]Phonetic Confidence:[/bold] {pkg.mean_confidence * 100:.1f}%\n")

    table = Table(title="Syllabic Ductus & Acoustic Formant Schedule", show_header=True)
    table.add_column("Syllable", style="bold yellow", width=10)
    table.add_column("Sign", style="bold cyan", width=8)
    table.add_column("IPA", style="bold green", width=14)
    table.add_column("Tier", justify="center", width=8)
    table.add_column("Weight", justify="center", width=8)
    table.add_column("F0 (Hz)", justify="right", width=9)
    table.add_column("Dur (ms)", justify="right", width=9)
    table.add_column("F1 / F2 (Hz)", justify="right", width=16)
    table.add_column("Manner", width=12)

    for s in pkg.syllables:
        tier_color = "green" if s.confidence_tier == "E4" else ("yellow" if s.confidence_tier == "E3" else ("dark_orange" if s.confidence_tier == "E2" else "red"))
        table.add_row(
            s.syllable,
            s.sign_code,
            s.ipa,
            f"[{tier_color}]{s.confidence_tier}[/{tier_color}]",
            f"{s.symbol} ({s.weight})",
            f"{s.f0_start_hz:.0f}",
            f"{s.duration_ms:.0f}",
            f"{s.formant_f1:.0f} / {s.formant_f2:.0f}",
            s.consonant_manner,
        )
    console.print(table)

    if export:
        out_p = Path(export)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(pkg.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n[green]Recitation package exported to {out_p}[/green]")


if __name__ == "__main__":
    app()

