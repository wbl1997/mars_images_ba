#!/usr/bin/env python3
"""Compare matching relationships stored in intra match text files."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


PLACEHOLDER_TOKEN = "REDi_i+1"
DEFAULT_BATCH_GLOB = "*_intra__.txt"


@dataclass(frozen=True)
class MatchRecord:
    line_no: int
    src_x: float
    src_y: float
    matched: bool
    img_id: int | None
    dst_x: float | None
    dst_y: float | None
    score: float | None
    raw: str


@dataclass
class ParseResult:
    path: Path
    records: dict[tuple[float, float], MatchRecord]
    duplicates: list[dict[str, object]]
    malformed: list[dict[str, object]]


@dataclass
class DiffResult:
    file_a: str
    file_b: str
    count_a: int
    count_b: int
    matched_a: int
    matched_b: int
    only_in_a: list[dict[str, object]]
    only_in_b: list[dict[str, object]]
    status_changed: list[dict[str, object]]
    image_id_changed: list[dict[str, object]]
    target_changed: list[dict[str, object]]
    score_changed: list[dict[str, object]]
    duplicates_a: list[dict[str, object]]
    duplicates_b: list[dict[str, object]]
    malformed_a: list[dict[str, object]]
    malformed_b: list[dict[str, object]]
    position_delta_stats: dict[str, object]


@dataclass
class BatchFileResult:
    name: str
    identical: bool
    diff: DiffResult


@dataclass
class BatchDiffResult:
    root_a: str
    root_b: str
    selection: str
    matched_files_a: int
    matched_files_b: int
    files_only_in_a: list[str]
    files_only_in_b: list[str]
    files: list[BatchFileResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "对比两个 intra 匹配文件中的匹配关系差异。"
            "既支持单文件，也支持目录/REDi_i+1 占位符批量对比。"
        )
    )
    parser.add_argument("path_a", type=Path, help="第一个文件、目录或占位符路径")
    parser.add_argument("path_b", type=Path, help="第二个文件、目录或占位符路径")
    parser.add_argument(
        "--pattern",
        help=(
            "目录批量对比时使用的文件名 glob。"
            f"默认自动推断；普通目录模式默认为 {DEFAULT_BATCH_GLOB!r}。"
        ),
    )
    parser.add_argument(
        "--coord-decimals",
        type=int,
        default=6,
        help="比较坐标时保留的小数位，默认 6。",
    )
    parser.add_argument(
        "--score-tol",
        type=float,
        default=1e-6,
        help="判定分数变化的阈值，默认 1e-6。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="摘要中每类信息最多展示多少条，默认 20。",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="将完整差异结果写入 JSON 文件。",
    )
    return parser


def round_value(value: float, decimals: int) -> float:
    return round(value, decimals)


def point_key(x: float, y: float, decimals: int) -> tuple[float, float]:
    return (round_value(x, decimals), round_value(y, decimals))


def format_point(x: float | None, y: float | None, decimals: int) -> str:
    if x is None or y is None:
        return "None"
    return f"({x:.{decimals}f}, {y:.{decimals}f})"


def format_record(record: MatchRecord, decimals: int) -> str:
    src = format_point(record.src_x, record.src_y, decimals)
    if not record.matched:
        return f"src={src} -> unmatched"
    dst = format_point(record.dst_x, record.dst_y, decimals)
    return f"src={src} -> dst={dst}, score={record.score:.6f}"


def parse_file(path: Path, decimals: int) -> ParseResult:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    records: dict[tuple[float, float], MatchRecord] = {}
    duplicates: list[dict[str, object]] = []
    malformed: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw = raw_line.rstrip("\n\r")
            stripped = raw.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) not in (3, 7):
                malformed.append(
                    {"line_no": line_no, "reason": f"字段数为 {len(parts)}，期望 3 或 7", "raw": raw}
                )
                continue

            try:
                src_x = float(parts[1])
                src_y = float(parts[2])
                matched = len(parts) == 7 and int(float(parts[0])) != 0
                img_id = int(float(parts[3])) if matched else None
                dst_x = float(parts[4]) if matched else None
                dst_y = float(parts[5]) if matched else None
                score = float(parts[6]) if matched else None
            except ValueError as exc:
                malformed.append({"line_no": line_no, "reason": str(exc), "raw": raw})
                continue

            key = point_key(src_x, src_y, decimals)
            record = MatchRecord(
                line_no=line_no,
                src_x=src_x,
                src_y=src_y,
                matched=matched,
                img_id=img_id,
                dst_x=dst_x,
                dst_y=dst_y,
                score=score,
                raw=raw,
            )

            if key in records:
                duplicates.append(
                    {
                        "point": {"x": key[0], "y": key[1]},
                        "first_line": records[key].line_no,
                        "duplicate_line": line_no,
                        "first_raw": records[key].raw,
                        "duplicate_raw": raw,
                    }
                )
                continue

            records[key] = record

    return ParseResult(path=path, records=records, duplicates=duplicates, malformed=malformed)


def calc_abs_delta_stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "variance": None,
        }

    abs_values = [abs(value) for value in values]
    count = len(abs_values)
    mean_value = sum(abs_values) / count
    max_value = max(abs_values)
    variance_value = sum((value - mean_value) ** 2 for value in abs_values) / count
    return {
        "count": count,
        "mean": mean_value,
        "max": max_value,
        "variance": variance_value,
    }


def compare_results(
    result_a: ParseResult, result_b: ParseResult, decimals: int, score_tol: float
) -> DiffResult:
    keys_a = set(result_a.records)
    keys_b = set(result_b.records)
    common_keys = sorted(keys_a & keys_b)

    only_in_a = [record_to_dict(result_a.records[key], decimals) for key in sorted(keys_a - keys_b)]
    only_in_b = [record_to_dict(result_b.records[key], decimals) for key in sorted(keys_b - keys_a)]

    status_changed: list[dict[str, object]] = []
    image_id_changed: list[dict[str, object]] = []
    target_changed: list[dict[str, object]] = []
    score_changed: list[dict[str, object]] = []
    delta_x_values: list[float] = []
    delta_y_values: list[float] = []

    for key in common_keys:
        record_a = result_a.records[key]
        record_b = result_b.records[key]
        src = {"x": key[0], "y": key[1]}

        if record_a.matched != record_b.matched:
            status_changed.append(
                {
                    "src": src,
                    "a": record_to_dict(record_a, decimals),
                    "b": record_to_dict(record_b, decimals),
                }
            )
            continue

        if not record_a.matched:
            continue

        if record_a.img_id != record_b.img_id:
            image_id_changed.append(
                {
                    "src": src,
                    "a": record_to_dict(record_a, decimals),
                    "b": record_to_dict(record_b, decimals),
                }
            )
            continue

        delta_x_values.append((record_b.dst_x or 0.0) - (record_a.dst_x or 0.0))
        delta_y_values.append((record_b.dst_y or 0.0) - (record_a.dst_y or 0.0))

        dst_a = point_key(record_a.dst_x or 0.0, record_a.dst_y or 0.0, decimals)
        dst_b = point_key(record_b.dst_x or 0.0, record_b.dst_y or 0.0, decimals)

        if dst_a != dst_b:
            target_changed.append(
                {
                    "src": src,
                    "a": record_to_dict(record_a, decimals),
                    "b": record_to_dict(record_b, decimals),
                }
            )
            continue

        score_a = record_a.score
        score_b = record_b.score
        if score_a is None or score_b is None:
            continue

        if abs(score_a - score_b) > score_tol:
            score_changed.append(
                {
                    "src": src,
                    "target": {"x": dst_a[0], "y": dst_a[1]},
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": score_b - score_a,
                }
            )

    position_delta_stats = {
        "matched_common_same_img_id": len(delta_x_values),
        "skipped_due_to_img_id_change": len(image_id_changed),
        "basis": "abs(B - A), only when src is same, both matched, and imgID is same",
        "x": calc_abs_delta_stats(delta_x_values),
        "y": calc_abs_delta_stats(delta_y_values),
    }

    return DiffResult(
        file_a=str(result_a.path),
        file_b=str(result_b.path),
        count_a=len(result_a.records),
        count_b=len(result_b.records),
        matched_a=sum(record.matched for record in result_a.records.values()),
        matched_b=sum(record.matched for record in result_b.records.values()),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        status_changed=status_changed,
        image_id_changed=image_id_changed,
        target_changed=target_changed,
        score_changed=score_changed,
        duplicates_a=result_a.duplicates,
        duplicates_b=result_b.duplicates,
        malformed_a=result_a.malformed,
        malformed_b=result_b.malformed,
        position_delta_stats=position_delta_stats,
    )


def record_to_dict(record: MatchRecord, decimals: int) -> dict[str, object]:
    return {
        "line_no": record.line_no,
        "src": {"x": round_value(record.src_x, decimals), "y": round_value(record.src_y, decimals)},
        "matched": record.matched,
        "img_id": record.img_id,
        "dst": None
        if not record.matched
        else {"x": round_value(record.dst_x or 0.0, decimals), "y": round_value(record.dst_y or 0.0, decimals)},
        "score": None if record.score is None else record.score,
        "raw": record.raw,
    }


def diff_has_difference(diff: DiffResult) -> bool:
    return any(
        (
            diff.only_in_a,
            diff.only_in_b,
            diff.status_changed,
            diff.image_id_changed,
            diff.target_changed,
            diff.score_changed,
            diff.duplicates_a,
            diff.duplicates_b,
            diff.malformed_a,
            diff.malformed_b,
        )
    )


def diff_change_count(diff: DiffResult) -> int:
    return sum(
        (
            len(diff.only_in_a),
            len(diff.only_in_b),
            len(diff.status_changed),
            len(diff.image_id_changed),
            len(diff.target_changed),
            len(diff.score_changed),
            len(diff.duplicates_a),
            len(diff.duplicates_b),
            len(diff.malformed_a),
            len(diff.malformed_b),
        )
    )


def build_name_matcher(
    pattern: str | None, template_names: tuple[str, str]
) -> tuple[Callable[[str], bool], str]:
    if pattern:
        return lambda name: fnmatch.fnmatch(name, pattern), f"glob:{pattern}"

    for template in template_names:
        if PLACEHOLDER_TOKEN not in template:
            continue

        regex_text = re.escape(template).replace(
            re.escape(PLACEHOLDER_TOKEN), r"RED\d+_\d+"
        )
        regex = re.compile(rf"^{regex_text}$")
        return lambda name, regex=regex: regex.fullmatch(name) is not None, f"template:{template}"

    return (
        lambda name: fnmatch.fnmatch(name, DEFAULT_BATCH_GLOB),
        f"glob:{DEFAULT_BATCH_GLOB}",
    )


def select_files(root: Path, matcher: Callable[[str], bool]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and matcher(path.name):
            files[path.name] = path
    return files


def compare_directories(
    root_a: Path,
    root_b: Path,
    matcher: Callable[[str], bool],
    selection: str,
    decimals: int,
    score_tol: float,
) -> BatchDiffResult:
    files_a = select_files(root_a, matcher)
    files_b = select_files(root_b, matcher)

    if not files_a and not files_b:
        raise FileNotFoundError(
            f"在两个目录中都没有找到符合条件的文件，选择规则为 {selection}"
        )

    names_a = set(files_a)
    names_b = set(files_b)
    common_names = sorted(names_a & names_b)

    files: list[BatchFileResult] = []
    for name in common_names:
        result_a = parse_file(files_a[name], decimals)
        result_b = parse_file(files_b[name], decimals)
        diff = compare_results(result_a, result_b, decimals, score_tol)
        files.append(BatchFileResult(name=name, identical=not diff_has_difference(diff), diff=diff))

    return BatchDiffResult(
        root_a=str(root_a),
        root_b=str(root_b),
        selection=selection,
        matched_files_a=len(files_a),
        matched_files_b=len(files_b),
        files_only_in_a=sorted(names_a - names_b),
        files_only_in_b=sorted(names_b - names_a),
        files=files,
    )


def print_summary(diff: DiffResult, decimals: int, limit: int) -> None:
    print("=== 文件概况 ===")
    print(f"A: {diff.file_a}")
    print(f"B: {diff.file_b}")
    print(f"A 记录数: {diff.count_a}, 其中匹配成功: {diff.matched_a}")
    print(f"B 记录数: {diff.count_b}, 其中匹配成功: {diff.matched_b}")
    print()

    print("=== 差异统计 ===")
    print(f"只在 A 中出现: {len(diff.only_in_a)}")
    print(f"只在 B 中出现: {len(diff.only_in_b)}")
    print(f"匹配状态变化: {len(diff.status_changed)}")
    print(f"匹配图像ID变化: {len(diff.image_id_changed)}")
    print(f"匹配目标变化: {len(diff.target_changed)}")
    print(f"分数变化: {len(diff.score_changed)}")
    print(f"A 重复源点: {len(diff.duplicates_a)}")
    print(f"B 重复源点: {len(diff.duplicates_b)}")
    print(f"A 异常行: {len(diff.malformed_a)}")
    print(f"B 异常行: {len(diff.malformed_b)}")
    print()

    print_position_delta_stats(diff.position_delta_stats)

    print_examples("只在 A 中出现", diff.only_in_a, limit, decimals)
    print_examples("只在 B 中出现", diff.only_in_b, limit, decimals)
    print_examples("匹配状态变化", diff.status_changed, limit, decimals)
    print_examples("匹配图像ID变化", diff.image_id_changed, limit, decimals)
    print_examples("匹配目标变化", diff.target_changed, limit, decimals)
    print_examples("分数变化", diff.score_changed, limit, decimals)
    print_examples("A 重复源点", diff.duplicates_a, limit, decimals)
    print_examples("B 重复源点", diff.duplicates_b, limit, decimals)
    print_examples("A 异常行", diff.malformed_a, limit, decimals)
    print_examples("B 异常行", diff.malformed_b, limit, decimals)


def format_stat_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def render_axis_delta_summary(axis_name: str, stats: dict[str, object]) -> str:
    return (
        f"{axis_name}: "
        f"count={format_stat_value(stats['count'])}, "
        f"mean={format_stat_value(stats['mean'])}, "
        f"max={format_stat_value(stats['max'])}, "
        f"variance={format_stat_value(stats['variance'])}"
    )


def print_position_delta_stats(position_delta_stats: dict[str, object]) -> None:
    print("=== 匹配位置差异统计 ===")
    print(f"统计基础: {position_delta_stats['basis']}")
    print(f"参与统计的共同成功匹配点数: {position_delta_stats['matched_common_same_img_id']}")
    print(f"因 imgID 不同而跳过的点数: {position_delta_stats['skipped_due_to_img_id_change']}")
    print(render_axis_delta_summary("x", position_delta_stats["x"]))
    print(render_axis_delta_summary("y", position_delta_stats["y"]))
    print()


def print_examples(title: str, items: list[dict[str, object]], limit: int, decimals: int) -> None:
    if not items:
        return

    print(f"=== {title}（最多显示 {min(limit, len(items))} 条） ===")
    for item in items[:limit]:
        print(render_item(item, decimals))
    if len(items) > limit:
        print(f"... 其余 {len(items) - limit} 条已省略")
    print()


def render_item(item: dict[str, object], decimals: int) -> str:
    if {"line_no", "src", "matched", "dst", "score", "raw"} <= item.keys():
        return render_record_dict(item, decimals)

    if {"src", "a", "b"} <= item.keys():
        src = item["src"]
        src_text = format_point(src["x"], src["y"], decimals)
        a_text = render_record_dict(item["a"], decimals)
        b_text = render_record_dict(item["b"], decimals)
        return f"src={src_text} | A: {a_text} | B: {b_text}"

    if {"src", "target", "score_a", "score_b", "delta"} <= item.keys():
        src = item["src"]
        target = item["target"]
        src_text = format_point(src["x"], src["y"], decimals)
        target_text = format_point(target["x"], target["y"], decimals)
        return (
            f"src={src_text} -> dst={target_text} | "
            f"score A={item['score_a']:.6f}, B={item['score_b']:.6f}, delta={item['delta']:.6f}"
        )

    if {"point", "first_line", "duplicate_line"} <= item.keys():
        point = item["point"]
        point_text = format_point(point["x"], point["y"], decimals)
        return f"src={point_text} | 首次出现在第 {item['first_line']} 行，重复出现在第 {item['duplicate_line']} 行"

    if {"line_no", "reason", "raw"} <= item.keys():
        return f"第 {item['line_no']} 行 | {item['reason']} | {item['raw']}"

    return json.dumps(item, ensure_ascii=False)


def render_record_dict(record: dict[str, object], decimals: int) -> str:
    src = record["src"]
    src_text = format_point(src["x"], src["y"], decimals)
    if not record["matched"]:
        return f"line={record['line_no']}, src={src_text} -> unmatched"

    dst = record["dst"]
    dst_text = format_point(dst["x"], dst["y"], decimals)
    return (
        f"line={record['line_no']}, src={src_text} -> "
        f"imgID={record['img_id']}, dst={dst_text}, score={record['score']:.6f}"
    )


def print_name_examples(title: str, names: list[str], limit: int) -> None:
    if not names:
        return

    print(f"=== {title}（最多显示 {min(limit, len(names))} 条） ===")
    for name in names[:limit]:
        print(name)
    if len(names) > limit:
        print(f"... 其余 {len(names) - limit} 条已省略")
    print()


def render_batch_summary(entry: BatchFileResult) -> str:
    diff = entry.diff
    position_delta_stats = diff.position_delta_stats
    x_stats = position_delta_stats["x"]
    y_stats = position_delta_stats["y"]
    return (
        f"{entry.name} | "
        f"matched_common_same_imgID={position_delta_stats['matched_common_same_img_id']}, "
        f"imgID_changed={position_delta_stats['skipped_due_to_img_id_change']}, "
        f"dx(mean/max/var)={format_stat_value(x_stats['mean'])}/"
        f"{format_stat_value(x_stats['max'])}/{format_stat_value(x_stats['variance'])}, "
        f"dy(mean/max/var)={format_stat_value(y_stats['mean'])}/"
        f"{format_stat_value(y_stats['max'])}/{format_stat_value(y_stats['variance'])} | "
        f"only_in_a={len(diff.only_in_a)}, only_in_b={len(diff.only_in_b)}, "
        f"status={len(diff.status_changed)}, imgID={len(diff.image_id_changed)}, "
        f"target={len(diff.target_changed)}, "
        f"score={len(diff.score_changed)}, dup_a={len(diff.duplicates_a)}, "
        f"dup_b={len(diff.duplicates_b)}, bad_a={len(diff.malformed_a)}, "
        f"bad_b={len(diff.malformed_b)}"
    )


def print_batch_summary(batch: BatchDiffResult, limit: int) -> None:
    differing_files = [entry for entry in batch.files if not entry.identical]
    identical_files = [entry.name for entry in batch.files if entry.identical]
    differing_files.sort(key=lambda entry: diff_change_count(entry.diff), reverse=True)

    print("=== 批量文件概况 ===")
    print(f"A 目录: {batch.root_a}")
    print(f"B 目录: {batch.root_b}")
    print(f"选择规则: {batch.selection}")
    print(f"A 命中文件数: {batch.matched_files_a}")
    print(f"B 命中文件数: {batch.matched_files_b}")
    print(f"共同文件数: {len(batch.files)}")
    print(f"完全一致文件: {len(identical_files)}")
    print(f"存在差异文件: {len(differing_files)}")
    print(f"只在 A 中出现的文件: {len(batch.files_only_in_a)}")
    print(f"只在 B 中出现的文件: {len(batch.files_only_in_b)}")
    print()

    print_name_examples("只在 A 中出现的文件", batch.files_only_in_a, limit)
    print_name_examples("只在 B 中出现的文件", batch.files_only_in_b, limit)

    if differing_files:
        print(f"=== 差异文件统计（最多显示 {min(limit, len(differing_files))} 条） ===")
        for entry in differing_files[:limit]:
            print(render_batch_summary(entry))
        if len(differing_files) > limit:
            print(f"... 其余 {len(differing_files) - limit} 条已省略")
        print()

    if identical_files:
        print_name_examples("完全一致的文件", identical_files, limit)


def diff_to_jsonable(diff: DiffResult) -> dict[str, object]:
    return asdict(diff)


def batch_to_jsonable(batch: BatchDiffResult) -> dict[str, object]:
    return asdict(batch)


def batch_has_difference(batch: BatchDiffResult) -> bool:
    return bool(
        batch.files_only_in_a
        or batch.files_only_in_b
        or any(not entry.identical for entry in batch.files)
    )


def run_single_file(
    path_a: Path,
    path_b: Path,
    decimals: int,
    score_tol: float,
    limit: int,
) -> tuple[int, dict[str, object]]:
    result_a = parse_file(path_a, decimals)
    result_b = parse_file(path_b, decimals)
    diff = compare_results(result_a, result_b, decimals, score_tol)
    print_summary(diff, decimals, limit)
    return (1 if diff_has_difference(diff) else 0, diff_to_jsonable(diff))


def run_batch_mode(
    root_a: Path,
    root_b: Path,
    matcher: Callable[[str], bool],
    selection: str,
    decimals: int,
    score_tol: float,
    limit: int,
) -> tuple[int, dict[str, object]]:
    batch = compare_directories(root_a, root_b, matcher, selection, decimals, score_tol)
    print_batch_summary(batch, limit)
    return (1 if batch_has_difference(batch) else 0, batch_to_jsonable(batch))


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.path_a.is_file() and args.path_b.is_file():
            exit_code, payload = run_single_file(
                args.path_a,
                args.path_b,
                args.coord_decimals,
                args.score_tol,
                args.limit,
            )
        else:
            if args.path_a.is_dir() and args.path_b.is_dir():
                root_a = args.path_a
                root_b = args.path_b
            elif args.path_a.parent.is_dir() and args.path_b.parent.is_dir() and (
                PLACEHOLDER_TOKEN in args.path_a.name or PLACEHOLDER_TOKEN in args.path_b.name
            ):
                root_a = args.path_a.parent
                root_b = args.path_b.parent
            else:
                missing = [str(path) for path in (args.path_a, args.path_b) if not path.exists()]
                if missing:
                    raise FileNotFoundError("路径不存在: " + ", ".join(missing))
                raise ValueError(
                    "请输入两个文件，两个目录，或两个带 REDi_i+1 占位符的路径。"
                )

            matcher, selection = build_name_matcher(
                args.pattern, (args.path_a.name, args.path_b.name)
            )
            exit_code, payload = run_batch_mode(
                root_a,
                root_b,
                matcher,
                selection,
                args.coord_decimals,
                args.score_tol,
                args.limit,
            )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"错误: {exc}")
        return 2

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"完整 JSON 已写入: {args.json_out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
