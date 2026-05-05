"""CLI: python -m sohn_translator <pdf> [--output-dir DIR] [--sheets SPREADSHEET_ID]"""
from __future__ import annotations
import asyncio
import logging
import sys
from pathlib import Path

import click

from .config import Config
from .pipeline import run_pipeline
from .writer import CSVWriter


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./out"),
    show_default=True,
    help="Directory for CSV output.",
)
@click.option(
    "--sheets",
    "sheets_id",
    type=str,
    default=None,
    help="Optional Google Sheets spreadsheet ID. Requires sohn-translator[sheets].",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip the LLM call. Exercises Pinecone retrieval and writers with stub rows.",
)
@click.option(
    "--no-retrieval",
    is_flag=True,
    default=False,
    help="Skip Pinecone+OpenAI embedding. Still calls the real Sohn LLM.",
)
def main(
    pdf_path: Path,
    output_dir: Path,
    sheets_id: str | None,
    log_level: str,
    dry_run: bool,
    no_retrieval: bool,
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.from_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    if sheets_id:
        from .writer import SheetsWriter  # lazy: needs google libs
        writer = SheetsWriter(cfg, spreadsheet_id=sheets_id)
    else:
        writer = CSVWriter(output_dir=output_dir)

    summary = asyncio.run(
        run_pipeline(
            cfg, pdf_path, writer, dry_run=dry_run, no_retrieval=no_retrieval
        )
    )
    click.echo("")
    click.echo("=" * 60)
    click.echo(f"Title:     {summary['title']}")
    click.echo(f"Session:   {summary['session_id']}")
    click.echo(f"Chunks:    {summary['num_chunks']}")
    click.echo(f"Rows:      {summary['num_rows']}")
    click.echo(f"Elapsed:   {summary['elapsed_s']}s")
    if isinstance(writer, CSVWriter):
        click.echo(f"Output:    {writer.path}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
