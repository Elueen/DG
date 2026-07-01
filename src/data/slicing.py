"""
src/data/slicing.py

预处理流水线 — 第三步A：轨迹切片（防泄露设计的"切片"半步）

设计原则（与 split.py 严格解耦）：
- 本模块只负责"把逐帧轨迹切成8秒窗口"，并记录每个窗口的精确时间范围
  和来源文件，**不在这里决定 train/val/test 归属**。
- 划分逻辑（哪些窗口属于train，哪些属于val/test）由 src/data/split.py
  独立负责。这样切片结果可以被多种不同的划分策略复用，不需要重新切片。
- 之所以能解耦，是因为切片输出本身已经携带了"这个窗口属于哪个源文件、
  哪个时间区间"的完整信息，划分阶段只需要读这张索引表做时间区间判断。

切片参数（固定，遵循NGSIM/highD轨迹预测任务的主流惯例）：
- 总窗口长度: 8秒 (3秒历史 + 5秒预测)
- 滑动步长: 1秒
- 在降采样后的5Hz数据上，8秒 = 40帧，1秒步长 = 5帧
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------------------------------------------------------------
# 切片参数常量
# ----------------------------------------------------------------------
HISTORY_SECONDS = 3.0
FUTURE_SECONDS = 5.0
WINDOW_SECONDS = HISTORY_SECONDS + FUTURE_SECONDS  # 8.0
STRIDE_SECONDS = 1.0


@dataclass
class WindowMetadata:
    """
    单个切片窗口的元数据。

    这是切片阶段的核心产出 —— 不存数值数据本身（数值数据按需在
    Dataset.__getitem__ 时从 parquet 里现读现切，避免内存爆炸），
    只存"这个窗口在哪"的索引信息。划分阶段只需要操作这张表。
    """

    window_id: str             # 全局唯一ID: f"{source_file}_{vehicle_id}_{start_frame}"
    source_file: str           # 来源文件名（不含路径），用于按文件做分层
    dataset_name: str          # "ngsim" / "highd"
    vehicle_id: int            # ego车辆ID
    start_frame: int           # 窗口起始帧（降采样后的frame序号，组内位置）
    end_frame: int             # 窗口结束帧（不含），= start_frame + n_frames_window
    start_time_sec: float      # 窗口起始时间（相对该车辆轨迹起点，秒）
    end_time_sec: float        # 窗口结束时间（秒）
    n_frames_hist: int         # 历史帧数（应=15，对应3秒*5Hz）
    n_frames_fut: int          # 预测帧数（应=25，对应5秒*5Hz）
    source_freq_hz: float      # 该窗口来源数据的原始采样频率（debug追溯用）
    target_freq_hz: float      # 降采样后的目标频率（应=5.0）


def _frames_for_duration(freq_hz: float, seconds: float) -> int:
    """把秒数换算成帧数，要求能整除（5Hz下8秒=40帧，1秒=5帧，均能整除）。"""
    raw = seconds * freq_hz
    n = round(raw)
    if abs(raw - n) > 1e-6:
        raise ValueError(
            f"时长 {seconds}秒 在频率 {freq_hz}Hz 下无法整除为整数帧 "
            f"(计算值={raw})。请检查降采样目标频率是否合理。"
        )
    return n


def slice_single_vehicle(
    vehicle_df: pd.DataFrame,
    source_file: str,
    dataset_name: str,
    vehicle_id: int,
    target_freq_hz: float,
    source_freq_hz: float,
    frame_col: str = "frame",
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    history_seconds: float = HISTORY_SECONDS,
) -> list[WindowMetadata]:
    """
    对单辆车的连续轨迹做滑动窗口切片。

    假设输入 vehicle_df 已经是单辆车、按 frame_col 排序、且帧是连续的
    （即降采样后组内"第几帧"严格递增，没有缺帧跳跃 —— 这个假设由调用方
    通过 frame 列的连续性检查来保证，见 slice_dataframe 中的处理）。

    Args:
        vehicle_df: 单辆车的逐帧数据（已降采样到target_freq_hz）
        source_file: 来源文件名，写入每个窗口的元数据
        dataset_name: "ngsim" 或 "highd"
        vehicle_id: 车辆ID
        target_freq_hz: 降采样后频率（用于换算窗口帧数）
        source_freq_hz: 原始频率（仅作debug记录，不参与计算）
        frame_col: 帧号列名
        window_seconds / stride_seconds / history_seconds: 切片参数

    Returns:
        该车辆产出的所有窗口元数据列表（可能为空，如果轨迹太短）
    """
    n_frames_window = _frames_for_duration(target_freq_hz, window_seconds)
    n_frames_stride = _frames_for_duration(target_freq_hz, stride_seconds)
    n_frames_hist = _frames_for_duration(target_freq_hz, history_seconds)
    n_frames_fut = n_frames_window - n_frames_hist

    n_total_frames = len(vehicle_df)
    dt = 1.0 / target_freq_hz

    windows = []
    # 滑动窗口起点：从0开始，每次走n_frames_stride，
    # 直到 start + n_frames_window <= n_total_frames（窗口必须完整，不允许末尾截断）
    start = 0
    while start + n_frames_window <= n_total_frames:
        end = start + n_frames_window  # 不含

        window = WindowMetadata(
            window_id=f"{source_file}_{vehicle_id}_{start}",
            source_file=source_file,
            dataset_name=dataset_name,
            vehicle_id=int(vehicle_id),
            start_frame=int(start),
            end_frame=int(end),
            start_time_sec=float(start * dt),
            end_time_sec=float((end - 1) * dt),  # 窗口最后一帧的时间（含）
            n_frames_hist=n_frames_hist,
            n_frames_fut=n_frames_fut,
            source_freq_hz=source_freq_hz,
            target_freq_hz=target_freq_hz,
        )
        windows.append(window)
        start += n_frames_stride

    return windows


def _check_frame_continuity(
    df: pd.DataFrame,
    vehicle_id_col: str,
    frame_col: str,
) -> dict[int, list[pd.DataFrame]]:
    """
    检查每辆车的frame是否连续（降采样后理论上应该是严格+1递增）。

    如果某辆车的轨迹中间有缺帧（比如车辆短暂被遮挡导致检测中断），
    直接对整段做滑动窗口会产生"看起来连续但实际跨越了缺失段"的窗口，
    这是一个常见但容易被忽视的数据质量问题。

    处理方式：把每辆车的轨迹按连续段拆分成多个子轨迹（sub-tracks），
    每个子轨迹内部 frame 严格连续，分别独立做切片。

    Returns:
        {vehicle_id: [sub_track_df1, sub_track_df2, ...]}
    """
    result = {}
    n_vehicles_with_gaps = 0

    for vid, group in df.groupby(vehicle_id_col):
        group_sorted = group.sort_values(frame_col).reset_index(drop=True)
        frames = group_sorted[frame_col].to_numpy()

        if len(frames) < 2:
            result[vid] = [group_sorted]
            continue

        gaps = np.diff(frames)
        # 降采样后，组内相邻行之间的frame差值理论上应该恒定
        # （取决于原始frame列的编号方式，这里不强行假设具体差值是多少，
        #  只检查"是否处处相等"，不相等的位置就是断点）
        mode_gap = pd.Series(gaps).mode().iloc[0]
        break_points = np.where(gaps != mode_gap)[0]  # gap[i]对应frames[i]到frames[i+1]

        if len(break_points) == 0:
            result[vid] = [group_sorted]
        else:
            n_vehicles_with_gaps += 1
            sub_tracks = []
            prev = 0
            for bp in break_points:
                sub_tracks.append(group_sorted.iloc[prev : bp + 1].reset_index(drop=True))
                prev = bp + 1
            sub_tracks.append(group_sorted.iloc[prev:].reset_index(drop=True))
            result[vid] = sub_tracks

    if n_vehicles_with_gaps > 0:
        logger.warning(
            f"{n_vehicles_with_gaps} 辆车的轨迹存在帧缺失（非连续），"
            f"已自动拆分为多段独立子轨迹分别切片，避免窗口跨越缺失段。"
        )

    return result


def slice_dataframe(
    df: pd.DataFrame,
    source_file: str,
    dataset_name: str,
    target_freq_hz: float,
    source_freq_hz: float,
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    history_seconds: float = HISTORY_SECONDS,
) -> pd.DataFrame:
    """
    对整个文件的所有车辆做滑动窗口切片，返回窗口索引表（DataFrame）。

    这是切片阶段的主入口。流程：
    1. 检查每辆车的帧连续性，缺帧处自动拆分为独立子轨迹
    2. 对每个（车辆, 子轨迹）独立调用 slice_single_vehicle
    3. 汇总所有窗口的元数据，返回一张表

    注意：本函数不读取/处理位置、速度等数值字段，只依赖 frame_col 和
    vehicle_id_col 来确定窗口边界。数值特征在 Dataset.__getitem__ 阶段
    按 (source_file, vehicle_id, start_frame, end_frame) 现读现切。

    Returns:
        DataFrame，每行是一个窗口的元数据，列对应 WindowMetadata 的字段
    """
    logger.info(f"开始对文件切片: {source_file} (dataset={dataset_name})")

    sub_tracks_by_vehicle = _check_frame_continuity(df, vehicle_id_col, frame_col)

    all_windows: list[WindowMetadata] = []
    n_vehicles_too_short = 0

    for vid, sub_tracks in sub_tracks_by_vehicle.items():
        for sub_df in sub_tracks:
            windows = slice_single_vehicle(
                sub_df,
                source_file=source_file,
                dataset_name=dataset_name,
                vehicle_id=vid,
                target_freq_hz=target_freq_hz,
                source_freq_hz=source_freq_hz,
                frame_col=frame_col,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                history_seconds=history_seconds,
            )
            if len(windows) == 0:
                n_vehicles_too_short += 1
            all_windows.extend(windows)

    if n_vehicles_too_short > 0:
        logger.info(
            f"{n_vehicles_too_short} 个(车辆/子轨迹)因长度不足8秒被跳过，未产生窗口。"
        )

    if len(all_windows) == 0:
        logger.warning(f"文件 {source_file} 没有产出任何有效窗口！请检查轨迹长度是否普遍过短。")
        return pd.DataFrame(columns=[f.name for f in WindowMetadata.__dataclass_fields__.values()])

    window_df = pd.DataFrame([w.__dict__ for w in all_windows])

    logger.info(
        f"文件 {source_file} 切片完成: {df[vehicle_id_col].nunique()} 辆车 → "
        f"{len(window_df)} 个窗口"
    )

    return window_df


def build_global_window_index(
    processed_files: list[tuple[str, str, float, float, str]],
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """
    批量处理多个文件，产出全局窗口索引表。

    这张表是后续 split.py 划分train/val/test的唯一输入，
    也是 Dataset 类按需读取数值数据的索引依据。

    Args:
        processed_files: 列表，每项为
            (parquet_path, dataset_name, target_freq_hz, source_freq_hz, source_file_tag)
            source_file_tag 用作window_id和后续分层划分的"文件"标识，
            通常用不含路径的文件名即可。

    Returns:
        全局窗口索引DataFrame，包含所有文件、所有车辆的窗口元数据
    """
    all_dfs = []
    for parquet_path, dataset_name, target_freq_hz, source_freq_hz, source_file_tag in processed_files:
        df = pd.read_parquet(parquet_path)
        window_df = slice_dataframe(
            df,
            source_file=source_file_tag,
            dataset_name=dataset_name,
            target_freq_hz=target_freq_hz,
            source_freq_hz=source_freq_hz,
            vehicle_id_col=vehicle_id_col,
            frame_col=frame_col,
        )
        all_dfs.append(window_df)

    global_index = pd.concat(all_dfs, ignore_index=True)

    logger.info(
        f"全局窗口索引构建完成: {len(processed_files)} 个文件 → "
        f"共 {len(global_index)} 个窗口"
    )
    _print_index_summary(global_index)

    return global_index


def _print_index_summary(window_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("全局窗口索引摘要")
    print("=" * 60)
    if len(window_df) == 0:
        print("⚠️ 索引为空！")
        print("=" * 60 + "\n")
        return

    print(f"总窗口数           : {len(window_df)}")
    print(f"涉及数据集         : {sorted(window_df['dataset_name'].unique())}")
    print(f"涉及来源文件数     : {window_df['source_file'].nunique()}")
    print("\n按数据集分组统计:")
    for name, sub in window_df.groupby("dataset_name"):
        print(f"  {name:<10}: {len(sub):>8} 窗口, "
              f"{sub['vehicle_id'].nunique():>6} 个唯一车辆ID（注意跨文件可能重号）")
    print("\n按来源文件分组统计:")
    for name, sub in window_df.groupby("source_file"):
        print(f"  {name:<45}: {len(sub):>8} 窗口")
    print("=" * 60 + "\n")


# ----------------------------------------------------------------------
# 命令行入口：对单个已降采样的parquet文件做切片，输出窗口索引
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="对降采样后的轨迹数据做滑动窗口切片")
    parser.add_argument("--input", type=str, required=True, help="降采样后的parquet文件路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="ngsim / highd")
    parser.add_argument("--target_freq", type=float, default=5.0)
    parser.add_argument("--source_freq", type=float, required=True, help="该文件对应的原始采样频率")
    parser.add_argument("--vehicle_id_col", type=str, default="id")
    parser.add_argument("--frame_col", type=str, default="frame")
    parser.add_argument("--output", type=str, default=None, help="窗口索引输出路径(parquet)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    df = pd.read_parquet(path)
    source_file_tag = path.stem  # 不含扩展名的文件名，作为来源标识

    window_df = slice_dataframe(
        df,
        source_file=source_file_tag,
        dataset_name=args.dataset_name,
        target_freq_hz=args.target_freq,
        source_freq_hz=args.source_freq,
        vehicle_id_col=args.vehicle_id_col,
        frame_col=args.frame_col,
    )

    _print_index_summary(window_df)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_parquet(out_path, index=False)
        logger.info(f"窗口索引已保存至: {out_path}")


if __name__ == "__main__":
    main()