"""CLI entry point following the lens family pattern.

    conversation-analyser <path> [--json] [--llm] [...]   # analyse (default)
    conversation-analyser serve [--host H] [--port P]      # run the HTTP API

Human-readable summary by default; `--json` emits the full ConversationAnalysis
to stdout (this is what auto-analyser consumes). Diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import DEFAULT_PORT, IDLE_GAP_MIN
from .manifest import MANIFEST
from .models import ConversationAnalysis
from .pipeline import ConversationAnalyser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _serve(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        print(json.dumps(MANIFEST, indent=2))
        return

    parser = argparse.ArgumentParser(
        prog="conversation-analyser",
        description="Analyse a human-AI conversation: analytics + critical-thinking taxonomy.",
    )
    parser.add_argument("file", type=Path, help="conversation file (.json/.txt/.md/.pdf)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="JSON output")
    parser.add_argument("--llm", action="store_true", help="add the taxonomy/critical-thinking tier (needs [llm] + ANTHROPIC_API_KEY)")
    parser.add_argument("--no-embeddings", action="store_true", help="skip prompt self-similarity")
    parser.add_argument(
        "--parse-mode",
        choices=("auto", "structured", "heuristic", "llm-segment"),
        default="auto",
    )
    parser.add_argument("--idle-gap", type=float, default=IDLE_GAP_MIN, help="sub-session split (minutes)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        result = ConversationAnalyser(idle_gap_min=args.idle_gap).analyse(
            args.file,
            llm=args.llm,
            with_embeddings=not args.no_embeddings,
            parse_mode=args.parse_mode,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(result.model_dump_json(indent=2))
        return

    _print_human(result)


def _print_human(result: ConversationAnalysis) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(file=sys.stdout)
    agg = result.aggregate
    a = agg.analytics

    console.print(
        f"[bold]Input:[/bold] {result.input}  "
        f"[bold]Format:[/bold] {result.format_detected} ({result.parse_mode})  "
        f"[bold]Sessions:[/bold] {result.session_count}  "
        f"[bold]LLM:[/bold] {'yes' if result.llm_used else 'no'}"
    )
    console.print(
        f"[bold]Turns:[/bold] {a.turn_count} "
        f"(human {a.human_turn_count}, ai {a.assistant_turn_count})  "
        f"[bold]Words:[/bold] {a.total_words}  "
        f"[bold]Questions:[/bold] {a.question_ratio:.0%}  "
        f"[bold]Pushback:[/bold] {a.pushback_count}"
    )

    if agg.critical_thinking is not None and agg.taxonomy is not None:
        ct = agg.critical_thinking
        console.print(
            f"[bold]Critical-thinking score:[/bold] {ct.score:.0f}/100  "
            f"[bold]Band:[/bold] {ct.band}  "
            f"[bold]Longest engaged chain:[/bold] {agg.taxonomy.longest_engaged_chain}"
        )
        table = Table(show_header=True, header_style="bold")
        for code in agg.taxonomy.label_counts:
            table.add_column(code, justify="right")
        table.add_row(*[str(v) for v in agg.taxonomy.label_counts.values()])
        console.print(table)
    else:
        console.print("[dim]Critical-thinking tier skipped (run with --llm).[/dim]")

    if result.notes:
        console.print(f"[dim]Notes: {', '.join(result.notes)}[/dim]")


def _serve(argv: list[str]) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="conversation-analyser serve")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("CONVERSATION_ANALYSER_PORT", str(DEFAULT_PORT)))
    )
    parser.add_argument(
        "--host", default=os.getenv("CONVERSATION_ANALYSER_HOST", "127.0.0.1")
    )
    args = parser.parse_args(argv)
    uvicorn.run("conversation_analyser.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
