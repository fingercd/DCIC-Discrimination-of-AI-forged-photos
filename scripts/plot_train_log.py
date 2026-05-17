"""
从 train_vlm 的控制台输出日志中解析 loss 和 grad_norm，并绘制曲线图。
用法:
  python scripts/plot_train_log.py [日志文件路径]
  python scripts/plot_train_log.py train_log.txt -o loss_curves.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_log(log_path: str | Path) -> tuple[list[float], list[float]]:
    """从日志中提取 loss 和 grad_norm。"""
    losses: list[float] = []
    grad_norms: list[float] = []
    log_path = Path(log_path)
    if not log_path.exists():
        return losses, grad_norms

    # 匹配 {'loss': '1.307', 'grad_norm': '0.94', ...} 或类似行
    pattern = re.compile(
        r"'loss':\s*'([^']*)',\s*'grad_norm':\s*'([^']*)'",
        re.IGNORECASE,
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                    grad_norms.append(float(m.group(2)))
                except ValueError:
                    continue
    return losses, grad_norms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取训练日志 txt，绘制 loss 与 grad_norm 曲线"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=str(ROOT / "train_log.txt"),
        help="训练日志 txt 路径（默认: 项目根目录 train_log.txt）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出图片路径（默认: 与日志同目录、同名的 .png）",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()
    losses, grad_norms = parse_log(log_path)

    if not losses:
        print(f"未在 {log_path} 中找到 loss/grad_norm 记录，请确认日志格式。", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请先安装 matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    steps = list(range(1, len(losses) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(steps, losses, color="C0", linewidth=0.8, label="loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, grad_norms, color="C2", linewidth=0.8, label="grad_norm")
    ax2.set_ylabel("Grad Norm")
    ax2.set_xlabel("Step")
    ax2.set_title("Gradient Norm")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = args.output
    if not out_path:
        out_path = log_path.with_suffix(".png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已解析 {len(losses)} 条记录，图像已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
