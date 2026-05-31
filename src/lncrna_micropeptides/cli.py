import argparse
import sys
from pathlib import Path

from lncrna_micropeptides.stages import (
    prepare_data,
    run_getorf,
    annotate_orfs,
    score_micropeptides,
)


def _run_stage(stage_main, args, extra_args=None):
    """
    Запускает stage.main(), прокидывая CLI аргументы через sys.argv.
    """
    argv = [sys.argv[0]]

    if args.config:
        argv += ["--config", str(args.config)]

    if extra_args:
        argv += extra_args

    sys.argv = argv
    stage_main()


def run_prepare(args):
    _run_stage(prepare_data.main, args)


def run_getorf_stage(args):
    _run_stage(run_getorf.main, args)


def run_annotate(args):
    _run_stage(annotate_orfs.main, args)


def run_score(args):
    extra = [
        "--top-n", str(args.top_n),
        "--top-source", args.top_source,
    ]
    if args.verbose:
        extra.append("--verbose")

    _run_stage(score_micropeptides.main, args, extra)


def build_parser():
    parser = argparse.ArgumentParser(
        description="lncRNA micropeptide discovery pipeline CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Run data preparation stage")
    p_prepare.add_argument("--config", type=Path, required=True)
    p_prepare.set_defaults(func=run_prepare)

    # getorf
    p_getorf = subparsers.add_parser("getorf", help="Run ORF detection")
    p_getorf.add_argument("--config", type=Path, required=True)
    p_getorf.set_defaults(func=run_getorf_stage)

    # annotate
    p_annotate = subparsers.add_parser("annotate", help="Annotate ORFs")
    p_annotate.add_argument("--config", type=Path, required=True)
    p_annotate.set_defaults(func=run_annotate)

    # score
    p_score = subparsers.add_parser("score", help="Score micropeptides")
    p_score.add_argument("--config", type=Path, required=True)
    p_score.add_argument("--top-n", type=int, default=100)
    p_score.add_argument(
        "--top-source",
        choices=["all", "representative", "best_per_transcript", "best_per_gene"],
        default="best_per_gene",
    )
    p_score.add_argument("--verbose", action="store_true")
    p_score.set_defaults(func=run_score)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()