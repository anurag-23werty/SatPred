"""``sathealth`` command line interface.

    sathealth download   fetch + verify the NASA SMAP/MSL dataset (idempotent, resumable)
    sathealth verify     re-check every raw file against its pinned SHA-256
    sathealth prepare    validate, split, scale and store the dataset
    sathealth report     write the Markdown data card
    sathealth evaluate   score detectors under the fixed evaluation protocol
    sathealth decide     run the decision layer (calibrate, fuse, alarm, events) on saved scores
    sathealth ablation   run every decision-layer variant and write docs/DECISION.md
    sathealth leaderboard  write the Markdown results table
    sathealth info       print the resolved config and where outputs go

Every command accepts ``--config`` and repeatable ``--set key.sub=value`` overrides.
Exit codes: 0 success, 1 data/integrity failure, 2 usage error.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sathealth import __version__
from sathealth.config import Config, load_config
from sathealth.data.download import download_dataset, load_manifest, verify_dataset
from sathealth.data.telemanom import TelemanomDataset
from sathealth.detectors.registry import BASELINE_NAMES, DETECTOR_NAMES
from sathealth.features.validation import validate_dataset
from sathealth.logging import configure_logging, get_logger
from sathealth.pipeline.ablation import write_ablation
from sathealth.pipeline.decide import decide
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.leaderboard import write_leaderboard
from sathealth.pipeline.prepare import (
    DataIntegrityError,
    DataQualityError,
    prepare_dataset,
    run_dir_for,
)
from sathealth.pipeline.report import render_data_card
from sathealth.utils.seed import seed_everything

log = get_logger("sathealth.cli")


def _cmd_download(cfg: Config, _: argparse.Namespace) -> int:
    d = cfg.download
    report = download_dataset(
        cfg.paths.raw_dir,
        load_manifest(d.manifest),
        max_workers=d.max_workers,
        retries=d.retries,
        timeout_s=d.timeout_s,
        backoff_s=d.backoff_s,
    )
    return 0 if report.ok else 1


def _cmd_verify(cfg: Config, _: argparse.Namespace) -> int:
    report = verify_dataset(
        cfg.paths.raw_dir, load_manifest(cfg.download.manifest), cfg.download.max_workers
    )
    for path, status in sorted(report.statuses.items()):
        if status.value != "ok":
            log.error("verification failed", path=path, status=status.value)
    log.info("verification finished", **report.summary())
    return 0 if report.ok else 1


def _cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    try:
        result = prepare_dataset(cfg, verify=not args.skip_verify, force=args.force)
    except (DataIntegrityError, DataQualityError, FileNotFoundError) as exc:
        log.error("prepare failed", reason=str(exc))
        return 1
    print(result.run_dir)
    return 0


def _cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    dataset = TelemanomDataset(cfg.paths.raw_dir, cfg.dataset)
    card = render_data_card(
        dataset, validate_dataset(dataset, cfg), load_manifest(cfg.download.manifest)
    )
    if args.out is None:
        print(card)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(card)
        log.info("wrote data card", path=args.out)
    return 0


_GROUPS = {
    "all": DETECTOR_NAMES,
    "baselines": BASELINE_NAMES,
    "deep": tuple(n for n in DETECTOR_NAMES if n not in BASELINE_NAMES),
}


def _detectors(names: str | list[str]) -> list[str]:
    """Expand group names (all, baselines, deep), keeping order and dropping repeats."""
    # argparse yields the bare default string "all" when no detector is given
    selected = [names] if isinstance(names, str) else names
    expanded = [d for name in selected for d in _GROUPS.get(name, (name,))]
    return list(dict.fromkeys(expanded))


def _cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    try:
        for name in _detectors(args.detectors):
            result = evaluate_detector(cfg, name, force=args.force, jobs=args.jobs)
            overall = result.summary["subsets"]["all"]
            print(
                f"{name:8s} composite_f1={overall['composite_f1']:.3f} "
                f"event_f1={overall['event']['f1']:.3f} {result.run_dir}"
            )
    except FileNotFoundError as exc:
        log.error("evaluate failed", reason=str(exc))
        return 1
    return 0


def _cmd_decide(cfg: Config, args: argparse.Namespace) -> int:
    try:
        result = decide(cfg, force=args.force)
    except FileNotFoundError as exc:
        log.error("decide failed", reason=str(exc))
        return 1
    overall = result.summary["subsets"]["all"]
    print(
        f"{result.name} composite_f1={overall['composite_f1']:.3f} "
        f"event_f1={overall['event']['f1']:.3f} "
        f"fp_events_per_1k={overall['fp_events_per_1k_steps']:.2f} "
        f"within_budget={result.summary['within_budget']} {result.run_dir}"
    )
    return 0


def _cmd_ablation(cfg: Config, args: argparse.Namespace) -> int:
    try:
        out = write_ablation(cfg, args.out, force=args.force)
    except FileNotFoundError as exc:
        log.error("ablation failed", reason=str(exc))
        return 1
    log.info("wrote ablation report", path=out)
    return 0


def _cmd_leaderboard(cfg: Config, args: argparse.Namespace) -> int:
    names = _detectors(args.detectors) + ([cfg.decision.name] if args.with_decision else [])
    try:
        out = write_leaderboard(cfg, names, args.out)
    except FileNotFoundError as exc:
        log.error("leaderboard failed", reason=str(exc))
        return 1
    log.info("wrote leaderboard", path=out)
    return 0


def _cmd_info(cfg: Config, _: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"version": __version__, "run_dir": str(run_dir_for(cfg)), "config": cfg.to_dict()},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=None, help="YAML config (default: built-in defaults)"
    )
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, repeatable",
    )
    common.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(
        prog="sathealth", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"sathealth {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", parents=[common], help="download and verify the dataset")
    sub.add_parser("verify", parents=[common], help="verify raw files against checksums")
    prep = sub.add_parser("prepare", parents=[common], help="validate, split, scale, store")
    prep.add_argument("--force", action="store_true", help="rebuild even if the run exists")
    prep.add_argument(
        "--skip-verify", action="store_true", help="skip the raw checksum gate (not recommended)"
    )
    rep = sub.add_parser("report", parents=[common], help="write the Markdown data card")
    rep.add_argument("--out", type=Path, default=None, help="output path (default: stdout)")
    choices = [*DETECTOR_NAMES, *_GROUPS]
    ev = sub.add_parser("evaluate", parents=[common], help="evaluate detectors")
    ev.add_argument(
        "detectors",
        nargs="*",
        choices=choices,
        default="all",  # a list default trips argparse's choices check for nargs="*"
        help="detectors or groups (all, baselines, deep) to run (default: all)",
    )
    ev.add_argument("--force", action="store_true", help="re-run even if results exist")
    ev.add_argument(
        "--jobs", type=int, default=1, help="channels evaluated in parallel (results unchanged)"
    )
    lb = sub.add_parser("leaderboard", parents=[common], help="write the results table")
    lb.add_argument("detectors", nargs="*", choices=choices, default="all")
    lb.add_argument("--out", type=Path, default=Path("docs/BASELINES.md"))
    lb.add_argument(
        "--with-decision", action="store_true", help="also list the configured decision pipeline"
    )
    dec = sub.add_parser("decide", parents=[common], help="run the decision layer")
    dec.add_argument("--force", action="store_true", help="re-run even if results exist")
    abl = sub.add_parser("ablation", parents=[common], help="run all decision variants")
    abl.add_argument("--out", type=Path, default=Path("docs/DECISION.md"))
    abl.add_argument("--force", action="store_true", help="re-run even if results exist")
    sub.add_parser("info", parents=[common], help="print resolved config and run directory")
    return parser


_COMMANDS = {
    "download": _cmd_download,
    "verify": _cmd_verify,
    "prepare": _cmd_prepare,
    "report": _cmd_report,
    "evaluate": _cmd_evaluate,
    "decide": _cmd_decide,
    "ablation": _cmd_ablation,
    "leaderboard": _cmd_leaderboard,
    "info": _cmd_info,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except (ValueError, OSError) as exc:  # pydantic.ValidationError subclasses ValueError
        print(f"sathealth: invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(args.log_level or cfg.log_level)
    seed_everything(cfg.seed)
    return _COMMANDS[args.command](cfg, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
