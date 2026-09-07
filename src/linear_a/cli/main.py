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

    # Check Ollama
    client = OllamaClient()
    is_online = client.is_available()
    models = client.list_models() if is_online else []
    models_str = ", ".join(models) if models else "[yellow]No models found or host offline[/yellow]"

    # Check Corpus
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


if __name__ == "__main__":
    app()
