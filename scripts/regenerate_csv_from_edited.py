"""
从已手动修改的 submission.csv 重新生成符合要求的 CSV（UTF-8 编码）。
保留你修改过的 explanation，同时修复编码问题。
可选：同步更新 inference_results.json，使 JSON 中的 explanation 与 CSV 一致。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============ 默认路径 ============
CONFIG = {
    "input_csv": str(ROOT / "data" / "submission.csv"),
    "output_csv": str(ROOT / "data" / "submission_fixed.csv"),
    "inference_json": str(ROOT / "data" / "inference_results.json"),
}


def _read_csv_with_encoding(path: str) -> list[dict]:
    """尝试多种编码读取 CSV，返回 [{"image_name", "label", "location", "explanation"}, ...]"""
    # 先尝试严格解码
    encodings = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("gbk", "strict"),
        ("gb18030", "strict"),
        ("cp936", "strict"),
        ("gbk", "replace"),  # Excel 中文版常用 GBK，遇坏字节则替换
        ("gb18030", "replace"),
    ]
    last_error = None
    for enc, err_mode in encodings:
        try:
            with open(path, "r", encoding=enc, errors=err_mode) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows and "image_name" in rows[0]:
                return rows
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue

    # 若都失败，用 utf-8 + replace 尽量恢复内容（坏字节替换为 �）
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows and "image_name" in rows[0]:
            print("  [提示] 使用 utf-8+replace 模式读取，部分损坏字节已替换", flush=True)
            return rows
    except Exception as e:
        last_error = e

    raise ValueError(f"无法读取 {path}，最后错误: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从已修改的 submission.csv 重新生成 UTF-8 编码的 CSV，保留你的 explanation 修改"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=CONFIG["input_csv"],
        help="已手动修改的 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CONFIG["output_csv"],
        help="输出的 UTF-8 CSV 路径（符合提交要求）",
    )
    parser.add_argument(
        "--update-json",
        action="store_true",
        help="同时用 CSV 中的 explanation 更新 inference_results.json",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=CONFIG["inference_json"],
        help="inference_results.json 路径（与 --update-json 配合）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="直接覆盖原 submission.csv（默认输出到 submission_fixed.csv）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.overwrite:
        output_path = input_path

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 读取 CSV（自动尝试多种编码）
    print(f"读取: {input_path}", flush=True)
    rows = _read_csv_with_encoding(str(input_path))
    print(f"  共 {len(rows)} 行", flush=True)

    # 写入 UTF-8 CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label", "location", "explanation"])
        for r in rows:
            image_name = r.get("image_name", "")
            label = r.get("label", 0)
            location = r.get("location", "")
            explanation = r.get("explanation", "")
            writer.writerow([image_name, label, location, explanation])

    print(f"已写入 UTF-8 CSV: {output_path}", flush=True)

    # 可选：更新 inference_results.json
    if args.update_json:
        json_path = Path(args.json_path)
        if not json_path.exists():
            print(f"  [跳过] JSON 不存在: {json_path}", flush=True)
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            by_name = {r["image_name"]: r for r in results if isinstance(r, dict) and "image_name" in r}
            updated = 0
            for r in rows:
                name = r.get("image_name", "")
                expl = r.get("explanation", "")
                if name in by_name:
                    by_name[name]["explanation"] = expl
                    updated += 1
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已更新 inference_results.json 中 {updated} 条 explanation", flush=True)


if __name__ == "__main__":
    main()
