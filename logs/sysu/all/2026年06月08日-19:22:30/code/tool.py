#!/usr/bin/env python3
"""Generate a Markdown summary for ReID experiment logs.

Usage:
    python generate_experiment_summary.py
    python generate_experiment_summary.py --logs-root logs --output experiment_log_summary.md
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(
    rf"R1\s*:\s*({FLOAT_RE})\s*;.*?"
    rf"R10\s*:\s*({FLOAT_RE})\s*;.*?"
    rf"R20\s*:\s*({FLOAT_RE})\s*;.*?"
    rf"mAP\s*:\s*({FLOAT_RE})\s*;.*?"
    rf"mINP\s*:\s*({FLOAT_RE})",
    re.IGNORECASE,
)
EPOCH_RE = re.compile(
    r"(?:^|[;|\s])Epoch\s*:\s*(\d+)|phase\s+\d+\s+epoch\s+(\d+)",
    re.IGNORECASE,
)


@dataclass
class Metric:
    epoch: int | None
    r1: float
    r10: float
    r20: float
    map: float
    minp: float


@dataclass
class RunSummary:
    group: str
    run: str
    method: str
    status: str
    best_epoch: int | None
    r1: float | None
    r10: float | None
    r20: float | None
    map: float | None
    minp: float | None
    best_map: float | None
    best_map_epoch: int | None
    short_params: str
    full_params: str
    log_path: Path


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escape = False

    for i, ch in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def extract_namespace_body(text: str) -> str | None:
    marker = "Namespace("
    start = text.find(marker)
    if start < 0:
        return None

    i = start + len(marker)
    body_start = i
    depth = 1
    quote: str | None = None
    escape = False

    while i < len(text):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
        else:
            if ch in {"'", '"'}:
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[body_start:i]
        i += 1
    return None


def parse_namespace(text: str) -> dict[str, Any]:
    body = extract_namespace_body(text)
    if not body:
        return {}

    args: dict[str, Any] = {}
    for item in split_top_level(body):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            args[key] = ast.literal_eval(value)
        except Exception:
            args[key] = value
    return args


def to_percent(value: float) -> float:
    return value * 100.0 if abs(value) <= 1.5 else value


def parse_metrics(text: str) -> list[Metric]:
    metrics: list[Metric] = []
    current_epoch: int | None = None

    for line in text.splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            raw_epoch = epoch_match.group(1) or epoch_match.group(2)
            current_epoch = int(raw_epoch)

        metric_match = METRIC_RE.search(line)
        if metric_match:
            r1, r10, r20, map_value, minp = (
                to_percent(float(value)) for value in metric_match.groups()
            )
            metrics.append(
                Metric(
                    epoch=current_epoch,
                    r1=r1,
                    r10=r10,
                    r20=r20,
                    map=map_value,
                    minp=minp,
                )
            )
    return metrics


def value_is_enabled(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def value_is_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def infer_method(args: dict[str, Any]) -> str:
    has_rank_hard = value_is_positive(args.get("rank_hard_weight"))
    if value_is_enabled(args.get("enable_rgmfd")):
        return "RG-MFD+RankHard" if has_rank_hard else "RG-MFD"
    if has_rank_hard:
        return "RankHard"
    return "Base/CRE"


def path_parts_for_log(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    try:
        log_index = parts.index("logs")
        dataset = parts[log_index + 1]
        mode = parts[log_index + 2]
        run = parts[log_index + 3]
        if run.startswith("trial") and len(parts) > log_index + 4:
            run = f"{run}/{parts[log_index + 4]}"
        return dataset, mode, run
    except (ValueError, IndexError):
        return "unknown", "unknown", path.parents[1].name if len(path.parents) > 1 else path.stem


def infer_group(path: Path, args: dict[str, Any]) -> str:
    dataset_from_path, mode_from_path, _ = path_parts_for_log(path)
    dataset = str(args.get("dataset") or dataset_from_path).lower()

    if dataset == "sysu":
        mode = str(args.get("search_mode") or mode_from_path)
        return f"SYSU {mode}"

    if dataset in {"regdb", "llcm"}:
        mode = str(args.get("test_mode") or mode_from_path)
        return f"{dataset.upper()} {mode}"

    return f"{dataset.upper()} {mode_from_path}"


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def short_pretrain(value: Any) -> str:
    if value in {None, "", "default"}:
        return "default"
    text = str(value)
    name = Path(text).name
    return name or text


def build_short_params(args: dict[str, Any]) -> str:
    pairs = [
        ("arch", "arch", None),
        ("lr", "lr", None),
        ("milestones", "ms", None),
        ("seed", "seed", None),
        ("stage1_epoch", "s1", None),
        ("stage2_epoch", "s2", None),
        ("cre_sample_rate", "cre_sr", "-"),
        ("enable_rgmfd", "rgmfd", "-"),
        ("rgmfd_start_epoch", "start", None),
        ("rgmfd_rel_weight", "rel", None),
        ("rgmfd_orth_weight", "orth", None),
        ("rgmfd_gate_reg_weight", "gate", None),
        ("rank_hard_start", "rank_start", None),
        ("rank_hard_weight", "rank_w", None),
        ("rank_hard_margin", "rank_m", None),
        ("rank_hard_topk", "rank_k", None),
        ("rank_pos_weight", "rank_pos", None),
    ]

    chunks: list[str] = []
    for key, label, missing in pairs:
        if key in args:
            chunks.append(f"{label}={format_value(args[key])}")
        elif missing is not None:
            chunks.append(f"{label}={missing}")

    chunks.append(f"pretrain={short_pretrain(args.get('model_path'))}")
    return "; ".join(chunks)


def build_full_params(args: dict[str, Any]) -> str:
    if not args:
        return "-"
    return "; ".join(f"{key}={format_value(value)}" for key, value in args.items())


def infer_status(metrics: list[Metric], args: dict[str, Any]) -> str:
    if not metrics:
        return "no_eval"

    epochs = [metric.epoch for metric in metrics if metric.epoch is not None]
    if not epochs:
        return "partial"

    stage2_epoch = args.get("stage2_epoch")
    if isinstance(stage2_epoch, int) and stage2_epoch > 0:
        return "done" if max(epochs) >= stage2_epoch - 1 else "partial"

    return "partial"


def summarize_log(path: Path) -> RunSummary:
    text = path.read_text(encoding="utf-8", errors="ignore")
    args = parse_namespace(text)
    metrics = parse_metrics(text)
    _, _, run = path_parts_for_log(path)

    best_r1_metric: Metric | None = max(metrics, key=lambda metric: metric.r1) if metrics else None
    best_map_metric: Metric | None = max(metrics, key=lambda metric: metric.map) if metrics else None

    return RunSummary(
        group=infer_group(path, args),
        run=run,
        method=infer_method(args),
        status=infer_status(metrics, args),
        best_epoch=best_r1_metric.epoch if best_r1_metric else None,
        r1=best_r1_metric.r1 if best_r1_metric else None,
        r10=best_r1_metric.r10 if best_r1_metric else None,
        r20=best_r1_metric.r20 if best_r1_metric else None,
        map=best_r1_metric.map if best_r1_metric else None,
        minp=best_r1_metric.minp if best_r1_metric else None,
        best_map=best_map_metric.map if best_map_metric else None,
        best_map_epoch=best_map_metric.epoch if best_map_metric else None,
        short_params=build_short_params(args),
        full_params=build_full_params(args),
        log_path=path,
    )


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def fmt_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_epoch(value: int | None) -> str:
    return "-" if value is None else str(value)


def fmt_best_map(value: float | None, epoch: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}@{fmt_epoch(epoch)}"


def group_sort_key(group: str) -> tuple[int, str]:
    preferred = [
        "SYSU all",
        "SYSU indoor",
        "REGDB t2v",
        "REGDB v2t",
        "LLCM t2v",
        "LLCM v2t",
    ]
    try:
        return preferred.index(group), group
    except ValueError:
        return len(preferred), group


def run_sort_key(run: RunSummary) -> tuple[bool, float, float, str]:
    return (
        run.r1 is not None,
        run.r1 if run.r1 is not None else -1.0,
        run.map if run.map is not None else -1.0,
        run.run,
    )


def append_best_table(lines: list[str], runs_by_group: dict[str, list[RunSummary]]) -> None:
    lines.extend(
        [
            "## 各数据集最佳结果",
            "",
            "| 数据集 | Run | 方法 | Best Epoch | Rank-1 | Rank-10 | Rank-20 | mAP | mINP | Best mAP/Epoch | 日志 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for group in sorted(runs_by_group, key=group_sort_key):
        candidates = [run for run in runs_by_group[group] if run.r1 is not None]
        if not candidates:
            continue
        best = sorted(candidates, key=run_sort_key, reverse=True)[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(group),
                    md_cell(best.run),
                    md_cell(best.method),
                    fmt_epoch(best.best_epoch),
                    fmt_num(best.r1),
                    fmt_num(best.r10),
                    fmt_num(best.r20),
                    fmt_num(best.map),
                    fmt_num(best.minp),
                    fmt_best_map(best.best_map, best.best_map_epoch),
                    f"`{display_path(best.log_path)}`",
                ]
            )
            + " |"
        )
    lines.append("")


def append_group_table(lines: list[str], group: str, runs: list[RunSummary], include_details: bool) -> None:
    lines.extend(
        [
            f"## {group}",
            "",
            "| Run | 方法 | 状态 | Best Epoch | Rank-1 | Rank-10 | Rank-20 | mAP | mINP | Best mAP/Epoch | 关键参数 | 日志 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )

    for run in sorted(runs, key=run_sort_key, reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(run.run),
                    md_cell(run.method),
                    run.status,
                    fmt_epoch(run.best_epoch),
                    fmt_num(run.r1),
                    fmt_num(run.r10),
                    fmt_num(run.r20),
                    fmt_num(run.map),
                    fmt_num(run.minp),
                    fmt_best_map(run.best_map, run.best_map_epoch),
                    md_cell(run.short_params),
                    f"`{display_path(run.log_path)}`",
                ]
            )
            + " |"
        )
    lines.append("")

    if include_details:
        lines.extend(["<details>", f"<summary>{group} 参数详情</summary>", ""])
        for run in sorted(runs, key=run_sort_key, reverse=True):
            lines.append(f"- `{md_cell(run.run)}`: {md_cell(run.full_params)}")
        lines.extend(["", "</details>", ""])


def build_markdown(runs: list[RunSummary], include_details: bool = True) -> str:
    runs_by_group: dict[str, list[RunSummary]] = defaultdict(list)
    for run in runs:
        runs_by_group[run.group].append(run)

    lines = [
        "# 实验日志对比汇总",
        "",
        f"- 生成时间：{dt.date.today().isoformat()}",
        "- 扫描范围：`logs/**/log/log.txt`",
        f"- 日志数量：{len(runs)}",
        "- 指标表中的数值为百分数；排序默认按 Best Rank-1 从高到低。",
        "- `Best mAP` 单独列出是为了发现 Rank-1 与 mAP 峰值不在同一 epoch 的情况。",
        "",
    ]

    append_best_table(lines, runs_by_group)

    for group in sorted(runs_by_group, key=group_sort_key):
        append_group_table(lines, group, runs_by_group[group], include_details)

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default="logs", type=Path, help="Root directory of experiment logs.")
    parser.add_argument(
        "--glob",
        default="**/log/log.txt",
        help="Glob pattern relative to --logs-root.",
    )
    parser.add_argument(
        "--output",
        default="experiment_log_summary.md",
        type=Path,
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Do not include the collapsible full-parameter sections.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_files = sorted(args.logs_root.glob(args.glob))
    runs = [summarize_log(path) for path in log_files]
    markdown = build_markdown(runs, include_details=not args.no_details)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output} from {len(runs)} log files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
