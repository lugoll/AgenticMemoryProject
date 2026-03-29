"""
Agentic Memory Project — pipeline entry point.

Run ingestion and/or question-answering for a named experiment variant.

Examples
--------
# Ingest documents for the baseline_dummy variant:
    python main.py --variant baseline_dummy --ingest

# Run QA for the baseline_dummy variant (requires prior ingestion):
    python main.py --variant baseline_dummy --qa

# Run both steps in sequence:
    python main.py --variant baseline_dummy --ingest --qa
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.settings import load_config
from src.telemetry.tracker import register_tracker


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion and/or QA pipeline for an experiment variant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        required=True,
        metavar="NAME",
        help="Experiment variant name as defined in unified_config.yaml.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the offline ingestion step (reads from data/raw/).",
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run the question-answering step (writes to evaluations/).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        metavar="PATH",
        help="Base directory for raw dataset files. Default: data/raw/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations"),
        metavar="PATH",
        help="Directory for QA result files. Default: evaluations/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a custom YAML config file. Defaults to src/config/unified_config.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.ingest and not args.qa:
        print("Nothing to do — pass --ingest, --qa, or both.", file=sys.stderr)
        return 1

    config = load_config(args.variant, config_path=args.config)

    # Boot telemetry so LiteLLM calls are tracked and written to evaluations/.
    register_tracker(
        log_level=config.telemetry.log_level,
        output_dir=Path(config.telemetry.output_dir),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.ingest:
        from src.pipelines.ingest import run_ingest
        run_ingest(config, data_dir=args.data_dir)

    if args.qa:
        from src.pipelines.qa import run_qa
        results_path = run_qa(config, data_dir=args.data_dir, output_dir=args.output_dir)
        print(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
