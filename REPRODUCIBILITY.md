# Reproducibility

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run linear-a status
```

Large SigLA/GORILA-derived census dumps and generated workbench HTML are excluded from the public tree ([NOTICE](NOTICE)). Rebuild locally only if you have lawful access to source dumps.

CI: GitHub Actions runs sync + ruff + pytest on push/PR.
