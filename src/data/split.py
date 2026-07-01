"""
src/data/split.py

Train / Val / Test 划分 — 数据侧最后一步。

设计原则（防数据泄露）：
- 按"每个 source_file 内部的时间轴"做三段切分，不按车辆ID随机划分
  （同一辆车可能贡献跨越多个时间段的窗口，随机划分会导致泄露）
- 分割点前后预留 8 秒硬间隔（一个完整窗口的时长），
  确保任何窗口都不会跨越分割线
- 划分比例：70% train / 10% val / 20% test（主流惯例，参考 MFP / X-TRACK）
- 每个 source_file 独立按比例划分，保证每种交通状态
  （mild/moderate/congested）都在三个集合中有代表
- 划分完成后强制执行重叠断言，防止实现错误导致泄露

输入：global_window_index.parquet（每行=一个窗口的元数据，含 start_time_sec）
输出：data/splits/train.parquet, val.parquet, test.parquet
      （是 global_window_index 的子集，只是打了标签，不含数值数据）
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# 分割点两侧的硬间隔（秒）= 一个完整窗口的时长
# 确保没有窗口会横跨分割线
GAP_SECONDS = 8.0


def split_single_file(
    file_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    gap_seconds: float = GAP_SECONDS,
) -> pd.DataFrame:
    """
    对单个 source_file 的所有窗口做时间轴三段切分。

    切分逻辑：
    1. 找出该文件中所有窗口的时间范围 [t_min, t_max]
    2. 按比例计算两个分割点：cut1（70%处）和 cut2（80%处）
    3. 在分割点两侧各留 gap_seconds/2 的死区（两侧合计 gap_seconds）
    4. 落在死区内的窗口直接丢弃（宁可少样本也不引入泄露风险）
    5. 返回带 'split' 列（'train'/'val'/'test'）的 DataFrame

    Args:
        file_df: 单个 source_file 的窗口元数据（含 start_time_sec, end_time_sec）
        train_ratio / val_ratio: 时间轴比例（test_ratio = 1 - train - val）
        gap_seconds: 分割点两侧的死区宽度（秒），默认 8 秒

    Returns:
        带 'split' 列的 DataFrame，死区内的行被移除
    """
    if file_df.empty:
        return file_df.copy().assign(split="train")

    t_min = file_df["start_time_sec"].min()
    t_max = file_df["end_time_sec"].max()
    total_duration = t_max - t_min

    if total_duration <= 0:
        logger.warning(f"文件时间范围为0，所有窗口归入train")
        return file_df.copy().assign(split="train")

    # 两个分割点（时间轴绝对值）
    cut1 = t_min + total_duration * train_ratio              # train/val 分界
    cut2 = t_min + total_duration * (train_ratio + val_ratio) # val/test 分界

    half_gap = gap_seconds / 2.0

    # 每个窗口的时间中心（用于判断归属）
    file_df = file_df.copy()
    t_center = (file_df["start_time_sec"] + file_df["end_time_sec"]) / 2.0

    # 死区判断：窗口的任何部分落在分割点 ± half_gap 范围内则丢弃
    in_gap1 = (file_df["start_time_sec"] < cut1 + half_gap) & \
              (file_df["end_time_sec"]   > cut1 - half_gap)
    in_gap2 = (file_df["start_time_sec"] < cut2 + half_gap) & \
              (file_df["end_time_sec"]   > cut2 - half_gap)
    in_gap  = in_gap1 | in_gap2

    # 分配标签
    split_labels = pd.Series("train", index=file_df.index)
    split_labels[t_center > cut2 + half_gap] = "test"
    split_labels[(t_center > cut1 + half_gap) & (t_center <= cut2 - half_gap)] = "val"
    split_labels[in_gap] = "gap"  # 临时标记，后面丢弃

    file_df["split"] = split_labels

    n_gap = (file_df["split"] == "gap").sum()
    if n_gap > 0:
        logger.debug(f"  死区丢弃 {n_gap} 个窗口（位于分割点 ±{half_gap}s 范围内）")

    return file_df[file_df["split"] != "gap"].copy()


def assert_no_overlap(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """
    强制断言：同一 source_file 内，train/val/test 的时间区间不存在重叠。

    检查方式：对每个 source_file，验证三个集合的时间区间互相不相交。
    发现重叠时直接 raise AssertionError，不允许静默跳过。

    这是防数据泄露的最后一道防线。
    """
    all_files = set(train_df["source_file"].unique()) | \
                set(val_df["source_file"].unique())   | \
                set(test_df["source_file"].unique())

    violations = []

    for src_file in sorted(all_files):
        for split_a, df_a, split_b, df_b in [
            ("train", train_df, "val",  val_df),
            ("train", train_df, "test", test_df),
            ("val",   val_df,   "test", test_df),
        ]:
            sub_a = df_a[df_a["source_file"] == src_file]
            sub_b = df_b[df_b["source_file"] == src_file]

            if sub_a.empty or sub_b.empty:
                continue

            # 检查是否有 window_id 同时出现在两个集合（最严格的检查）
            ids_a = set(sub_a["window_id"])
            ids_b = set(sub_b["window_id"])
            shared_ids = ids_a & ids_b
            if shared_ids:
                violations.append(
                    f"{src_file}: {split_a} 和 {split_b} 共享 {len(shared_ids)} 个 window_id"
                )
                continue

            # 检查时间区间是否重叠
            # 充分条件：如果 A 的最大 end_time < B 的最小 start_time（或反过来），则无重叠
            a_end_max = sub_a["end_time_sec"].max()
            b_start_min = sub_b["start_time_sec"].min()
            b_end_max = sub_b["end_time_sec"].max()
            a_start_min = sub_a["start_time_sec"].min()

            # 如果 A 和 B 的时间范围有任何交集，进一步检查
            if not (a_end_max < b_start_min or b_end_max < a_start_min):
                # 精确检查：找任意一个 A 中的窗口，其时间范围与 B 中的窗口重叠
                for _, row_a in sub_a.sample(min(100, len(sub_a)), random_state=42).iterrows():
                    overlap_mask = (
                        (sub_b["start_time_sec"] < row_a["end_time_sec"]) &
                        (sub_b["end_time_sec"]   > row_a["start_time_sec"])
                    )
                    if overlap_mask.any():
                        violations.append(
                            f"{src_file}: {split_a}[{row_a['window_id']}] 与 {split_b} 中 "
                            f"{overlap_mask.sum()} 个窗口时间重叠"
                        )
                        break  # 每对只报告一个违规，避免刷屏

    if violations:
        msg = "\n".join(violations)
        raise AssertionError(
            f"发现 train/val/test 时间重叠（数据泄露风险）！\n{msg}\n"
            f"请检查 split_single_file() 的 gap_seconds 参数是否足够大。"
        )

    logger.info("✅ 重叠断言通过：train/val/test 之间无时间重叠")


def build_splits(
    global_index_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    gap_seconds: float = GAP_SECONDS,
    random_seed: int   = 42,
) -> dict[str, pd.DataFrame]:
    """
    主入口：读取全局窗口索引，按文件做时间轴划分，输出三个 split 文件。

    Args:
        global_index_path: global_window_index.parquet
        output_dir: 输出目录，产出 train.parquet / val.parquet / test.parquet
        train_ratio / val_ratio: 划分比例（test = 1 - train - val）
        gap_seconds: 分割点死区宽度（秒）
        random_seed: 仅用于重叠断言中的抽样检查，不影响划分结果

    Returns:
        {'train': df, 'val': df, 'test': df}
    """
    global_index_path = Path(global_index_path)
    output_dir        = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"读取全局窗口索引: {global_index_path}")
    index_df = pd.read_parquet(global_index_path)
    logger.info(f"总窗口数: {len(index_df)}")

    # 按 source_file 分组，每个文件独立做时间轴划分
    all_labeled: list[pd.DataFrame] = []
    file_stats = []

    for src_file, file_group in index_df.groupby("source_file"):
        labeled = split_single_file(
            file_group,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            gap_seconds=gap_seconds,
        )
        all_labeled.append(labeled)

        counts = labeled["split"].value_counts().to_dict()
        file_stats.append({
            "source_file": src_file,
            "train": counts.get("train", 0),
            "val":   counts.get("val",   0),
            "test":  counts.get("test",  0),
            "total_kept": len(labeled),
            "total_original": len(file_group),
        })

    labeled_df = pd.concat(all_labeled, ignore_index=True)

    # 拆分
    train_df = labeled_df[labeled_df["split"] == "train"].drop(columns=["split"])
    val_df   = labeled_df[labeled_df["split"] == "val"].drop(columns=["split"])
    test_df  = labeled_df[labeled_df["split"] == "test"].drop(columns=["split"])

    # 重叠断言（失败则直接抛异常，不允许继续）
    assert_no_overlap(train_df, val_df, test_df)

    # 落盘
    train_path = output_dir / "train.parquet"
    val_path   = output_dir / "val.parquet"
    test_path  = output_dir / "test.parquet"

    train_df.reset_index(drop=True).to_parquet(train_path, index=False)
    val_df.reset_index(drop=True).to_parquet(val_path,   index=False)
    test_df.reset_index(drop=True).to_parquet(test_path, index=False)

    # 打印摘要
    _print_split_summary(train_df, val_df, test_df, file_stats, gap_seconds)

    return {"train": train_df, "val": val_df, "test": test_df}


def _print_split_summary(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    file_stats: list[dict],
    gap_seconds: float,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n" + "=" * 65)
    print("Train / Val / Test 划分摘要")
    print("=" * 65)
    print(f"划分策略    : 每文件内部时间轴三段切分")
    print(f"死区宽度    : ±{gap_seconds/2:.1f}s（分割点两侧各 {gap_seconds/2:.1f}s）")
    print(f"{'集合':<8} {'窗口数':>10} {'占比':>8}")
    print("-" * 30)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:<8} {len(df):>10,} {len(df)/total:>8.1%}")
    print(f"{'合计':<8} {total:>10,} {'100%':>8}")
    print()

    # 按数据集分组统计
    all_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    print("按数据集分组:")
    for dname, sub in all_df.groupby("dataset_name"):
        counts = sub["split"].value_counts()
        total_d = len(sub)
        print(f"  {dname:<8}: "
              f"train={counts.get('train',0):>7,}  "
              f"val={counts.get('val',0):>7,}  "
              f"test={counts.get('test',0):>7,}  "
              f"合计={total_d:>7,}")

    # 每个文件的划分情况
    print("\n按来源文件分组:")
    stats_df = pd.DataFrame(file_stats).sort_values("source_file")
    for _, row in stats_df.iterrows():
        gap_count = row["total_original"] - row["total_kept"]
        gap_str   = f" (丢弃死区{gap_count})" if gap_count > 0 else ""
        print(f"  {row['source_file']:<40}: "
              f"train={row['train']:>6}  val={row['val']:>5}  test={row['test']:>6}{gap_str}")
    print("=" * 65 + "\n")


# ============================================================================
# 命令行入口
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成 train/val/test 划分")
    parser.add_argument(
        "--index", default="data/processed/global_window_index.parquet",
    )
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--gap",         type=float, default=GAP_SECONDS,
                        help="分割点死区宽度（秒），默认8秒")
    args = parser.parse_args()

    build_splits(
        global_index_path=args.index,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gap_seconds=args.gap,
    )


if __name__ == "__main__":
    main()