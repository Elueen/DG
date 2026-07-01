"""
scripts/batch_preprocess.py

批量预处理入口：对所有 NGSIM 和 highD 文件一次性完成：
    1. 平滑 → 降采样到5Hz → 重新差分（preprocess.py）
    2. 滑动窗口切片，产出窗口索引（slicing.py）
    3. 合并所有文件的窗口索引，产出一张全局索引表（供 split.py 划分用）

设计原则：
- 每个文件独立处理，失败不影响其他文件（try/except 隔离）
- 已处理过的文件自动跳过（--resume 模式），方便断点续跑
- 进度实时打印，方便在 SLURM 日志里追踪
- 全局索引表落盘在 data/processed/global_window_index.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

# 把项目根目录加进 sys.path，让 src.* 可以正常 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import run_smoothing_and_downsampling_pipeline
from src.data.slicing import slice_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 文件清单定义
# 格式：(原始csv路径, dataset_name, source_freq_hz, 输出文件标签)
# source_freq_hz 已通过物理一致性反推确认：NGSIM=10Hz, highD=25Hz
# ----------------------------------------------------------------------
def build_file_list(
    ngsim_dir: Path,
    highd_dir: Path,
    highd_n_files: int = 60,
) -> list[tuple[Path, str, float, str]]:
    """
    构建待处理文件清单。

    NGSIM 文件按硬编码路径列出（文件数量固定，路径明确）。
    highD 文件按编号 01~N 自动枚举（避免硬编码60次）。

    Returns:
        list of (raw_csv_path, dataset_name, source_freq_hz, file_tag)
        file_tag 用作输出文件名的前缀，也写入窗口索引的 source_file 字段
    """
    file_list: list[tuple[Path, str, float, str]] = []

    # ---- NGSIM US-101（3个15分钟时段，均已转换为highD格式，10Hz）----
    ngsim_us101_files = [
        "us-101/track_trajectories-0750-0805.csv",
        "us-101/track_trajectories-0805-0820.csv",
        "us-101/track_trajectories-0820-0835.csv",
    ]
    for rel_path in ngsim_us101_files:
        p = ngsim_dir / rel_path
        tag = "ngsim_us101_" + Path(rel_path).stem.replace("track_trajectories-", "")
        file_list.append((p, "ngsim", 10.0, tag))

    # ---- NGSIM I-80（3个15分钟时段，10Hz）----
    ngsim_i80_files = [
        "i-80/track_trajectories-0400-0415.csv",
        "i-80/track_trajectories-0500-0515.csv",
        "i-80/track_trajectories-0515-0530.csv",
    ]
    for rel_path in ngsim_i80_files:
        p = ngsim_dir / rel_path
        tag = "ngsim_i80_" + Path(rel_path).stem.replace("track_trajectories-", "")
        file_list.append((p, "ngsim", 10.0, tag))

    # ---- highD（01~60，25Hz）----
    for i in range(1, highd_n_files + 1):
        fname = f"{i:02d}_tracks.csv"
        p = highd_dir / fname
        tag = f"highd_{i:02d}"
        file_list.append((p, "highd", 25.0, tag))

    return file_list


def process_one_file(
    raw_csv: Path,
    dataset_name: str,
    source_freq_hz: float,
    file_tag: str,
    processed_dir: Path,
    target_freq_hz: float = 5.0,
    resume: bool = True,
) -> pd.DataFrame | None:
    """
    对单个原始CSV文件完成：预处理 → 切片 → 返回窗口索引DataFrame。

    Args:
        resume: 若为True，且目标parquet已存在，则跳过预处理直接做切片
                （切片索引文件若也存在，则完全跳过，直接读缓存）

    Returns:
        该文件产出的窗口索引DataFrame，失败时返回None
    """
    # 输出路径定义
    processed_parquet = processed_dir / dataset_name / f"{file_tag}_5hz.parquet"
    window_index_parquet = processed_dir / dataset_name / f"{file_tag}_windows.parquet"

    processed_parquet.parent.mkdir(parents=True, exist_ok=True)

    # ---- 完全缓存命中：直接读窗口索引，跳过一切 ----
    if resume and window_index_parquet.exists():
        logger.info(f"[SKIP] {file_tag} 窗口索引已存在，直接读取缓存")
        return pd.read_parquet(window_index_parquet)

    # ---- 检查原始文件是否存在 ----
    if not raw_csv.exists():
        logger.warning(f"[MISSING] 原始文件不存在，跳过: {raw_csv}")
        return None

    logger.info(f"[START] {file_tag} ({dataset_name}, {source_freq_hz}Hz → {target_freq_hz}Hz)")

    # ---- Step 1: 预处理（平滑→降采样→重新差分）----
    if resume and processed_parquet.exists():
        logger.info(f"  [SKIP preprocess] 已有降采样parquet，直接读取")
        df_processed = pd.read_parquet(processed_parquet)
    else:
        logger.info(f"  [preprocess] 读取 {raw_csv.name}")
        df_raw = pd.read_csv(raw_csv)
        logger.info(f"  原始数据: {df_raw.shape[0]} 行, {df_raw.shape[1]} 列")

        df_processed, pipeline_info = run_smoothing_and_downsampling_pipeline(
            df_raw,
            source_freq_hz=source_freq_hz,
            target_freq_hz=target_freq_hz,
            dataset_name=dataset_name,
        )
        df_processed.to_parquet(processed_parquet, index=False)
        logger.info(f"  [preprocess done] 降采样后 {len(df_processed)} 行，已存 {processed_parquet.name}")

    # ---- Step 2: 切片 ----
    logger.info(f"  [slicing] 开始切片...")
    window_df = slice_dataframe(
        df_processed,
        source_file=file_tag,
        dataset_name=dataset_name,
        target_freq_hz=target_freq_hz,
        source_freq_hz=source_freq_hz,
    )
    window_df.to_parquet(window_index_parquet, index=False)
    logger.info(f"  [slicing done] {len(window_df)} 个窗口，已存 {window_index_parquet.name}")

    return window_df


def main():
    parser = argparse.ArgumentParser(description="批量预处理所有NGSIM和highD文件")
    parser.add_argument(
        "--ngsim_dir", type=str,
        default="data/raw/ngsim",
        help="NGSIM原始文件根目录",
    )
    parser.add_argument(
        "--highd_dir", type=str,
        default="data/raw/highD",
        help="highD原始文件目录",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/processed",
        help="处理结果输出根目录",
    )
    parser.add_argument(
        "--highd_n_files", type=int, default=60,
        help="highD文件总数（默认60，即01~60）",
    )
    parser.add_argument(
        "--target_freq", type=float, default=5.0,
        help="目标降采样频率（Hz）",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="跳过已处理完成的文件（断点续跑，默认开启）",
    )
    parser.add_argument(
        "--no_resume", dest="resume", action="store_false",
        help="强制重新处理所有文件，忽略缓存",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "ngsim", "highd"],
        help="只处理指定数据集（调试用）",
    )
    args = parser.parse_args()

    ngsim_dir = Path(args.ngsim_dir)
    highd_dir = Path(args.highd_dir)
    processed_dir = Path(args.output_dir)

    # ---- 构建文件清单 ----
    file_list = build_file_list(ngsim_dir, highd_dir, args.highd_n_files)

    # 按 --dataset 过滤
    if args.dataset == "ngsim":
        file_list = [(p, d, f, t) for p, d, f, t in file_list if d == "ngsim"]
    elif args.dataset == "highd":
        file_list = [(p, d, f, t) for p, d, f, t in file_list if d == "highd"]

    logger.info(f"待处理文件总数: {len(file_list)}")
    logger.info(f"断点续跑模式: {'开启' if args.resume else '关闭（强制重处理）'}")

    # ---- 批量处理 ----
    all_window_dfs = []
    success_count = 0
    skip_count = 0
    fail_count = 0
    fail_files = []

    for i, (raw_csv, dataset_name, source_freq_hz, file_tag) in enumerate(file_list, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(file_list)}] {file_tag}")
        logger.info(f"{'='*60}")

        try:
            window_df = process_one_file(
                raw_csv=raw_csv,
                dataset_name=dataset_name,
                source_freq_hz=source_freq_hz,
                file_tag=file_tag,
                processed_dir=processed_dir,
                target_freq_hz=args.target_freq,
                resume=args.resume,
            )

            if window_df is None:
                fail_count += 1
                fail_files.append(file_tag)
            elif "[SKIP]" in logging.getLogger().handlers[0].format(
                logging.LogRecord("", 0, "", 0, "", [], None)
            ) if logging.getLogger().handlers else False:
                skip_count += 1
                all_window_dfs.append(window_df)
            else:
                success_count += 1
                all_window_dfs.append(window_df)

        except Exception:
            fail_count += 1
            fail_files.append(file_tag)
            logger.error(f"[FAIL] {file_tag} 处理失败:\n{traceback.format_exc()}")
            # 继续处理下一个文件，不中断整个批任务

    # ---- 合并全局窗口索引 ----
    logger.info(f"\n{'='*60}")
    logger.info("合并全局窗口索引...")
    logger.info(f"{'='*60}")

    if len(all_window_dfs) == 0:
        logger.error("没有任何文件处理成功，无法生成全局索引！")
        sys.exit(1)

    global_index = pd.concat(all_window_dfs, ignore_index=True)
    global_index_path = processed_dir / "global_window_index.parquet"
    global_index.to_parquet(global_index_path, index=False)

    # ---- 最终摘要 ----
    logger.info(f"\n{'='*60}")
    logger.info("批量处理完成摘要")
    logger.info(f"{'='*60}")
    logger.info(f"成功/跳过  : {success_count + (len(all_window_dfs) - success_count)}")
    logger.info(f"失败       : {fail_count}")
    if fail_files:
        logger.warning(f"失败文件列表: {fail_files}")
    logger.info(f"全局窗口总数: {len(global_index)}")
    logger.info(f"\n按数据集分组:")
    for name, sub in global_index.groupby("dataset_name"):
        logger.info(
            f"  {name:<10}: {len(sub):>8} 窗口, "
            f"{sub['vehicle_id'].nunique():>6} 个车辆ID, "
            f"{sub['source_file'].nunique():>3} 个文件"
        )
    logger.info(f"\n全局索引已保存: {global_index_path}")


if __name__ == "__main__":
    main()