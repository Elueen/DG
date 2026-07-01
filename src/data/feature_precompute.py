"""
src/data/feature_precompute.py

预计算脚本：对每个8秒窗口预算邻车选择（版本A：固定6辆社交网格）
以及TTC、THW、变道标注，结果缓存到 features_cache.parquet。

设计原则：
- 不依赖原始数据里的 leftPrecedingId 等字段（经验证不可靠）
- 完全从 (x, y, xVelocity, laneId) 坐标重新计算邻车关系
- highD 双向行驶：先按行驶方向分组，只在同方向车辆中选邻车
- NGSIM 单向行驶：所有车辆同方向，直接按 laneId/x 分配槽位
- TTC 无效时用 np.nan + 单独的 valid_ttc_mask，不用哨兵值
- 输出一个 parquet，window_id 为主键，Dataset.__getitem__ 按需读取
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 社交网格6个槽位的定义（相对ego的位置）
# 槽位顺序固定，Dataset 侧按此顺序解包
SLOT_NAMES = [
    "left_preceding",    # 左前
    "left_alongside",    # 左侧
    "left_following",    # 左后
    "right_preceding",   # 右前
    "right_alongside",   # 右侧
    "right_following",   # 右后
]
N_SLOTS = len(SLOT_NAMES)  # 6

LANE_WIDTH_DEFAULT = 3.5   # 米，用于变道判断阈值
SAME_LANE_Y_TOL   = 1.8    # 米，判断"同车道"的 y 偏差容限（半车道宽）


# ============================================================================
# 核心工具函数
# ============================================================================

def _detect_driving_direction(group_vx: pd.Series) -> int:
    """
    判断一组车辆的行驶方向。
    返回 +1（x增大方向）或 -1（x减小方向）。
    用速度中位数判断，比单辆车更稳健。
    """
    return 1 if group_vx.median() >= 0 else -1


def _split_by_direction(frame_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    把同一帧的所有车辆按行驶方向分成若干组。
    highD 双向：分成正向组和负向组。
    NGSIM 单向：只有一组。

    判断依据：xVelocity 的符号。
    边界情况：速度接近0的车（如停车）暂归入最近的方向组。
    """
    if frame_df.empty:
        return []

    # 用速度符号分组
    pos_mask = frame_df["xVelocity"] >= 0
    groups = []
    for mask in [pos_mask, ~pos_mask]:
        sub = frame_df[mask]
        if len(sub) >= 1:
            groups.append(sub)
    return groups


def _assign_slots_for_ego(
    ego_row: pd.Series,
    same_dir_frame: pd.DataFrame,
) -> dict:
    """
    对单辆 ego 车，在同方向车辆中分配6个社交网格槽位。

    槽位分配逻辑：
    1. 计算每辆他车相对 ego 的 dy（横向偏移）和 dx（纵向偏移，已考虑行驶方向）
    2. dy > SAME_LANE_Y_TOL：右侧车道
       dy < -SAME_LANE_Y_TOL：左侧车道
       |dy| <= SAME_LANE_Y_TOL：同车道
    3. 纵向上：前方(dx>0)、侧方(|dx|小)、后方(dx<0)
    4. 每个槽位只取最近的一辆车（按纵向距离排序取第一）

    注意：这里的"左/右"是相对行驶方向定义的：
    - NGSIM（x增大方向行驶）：y增大 = 右侧
    - highD 负向车辆（x减小方向行驶）：y增大 = 左侧（镜像）
    所以需要用 driving_direction 做符号修正。
    """
    ego_x  = ego_row["x"]
    ego_y  = ego_row["y"]
    ego_id = ego_row["id"]
    drv_dir = ego_row.get("_drv_dir", 1)  # +1 or -1

    # 排除自身
    others = same_dir_frame[same_dir_frame["id"] != ego_id].copy()
    if others.empty:
        return _empty_slots()

    # 相对坐标（纵向按行驶方向修正符号，横向按行驶方向做镜像修正）
    # dx > 0 表示"在ego前方"（沿行驶方向）
    others["_dx"] = (others["x"] - ego_x) * drv_dir
    # dy > 0 表示"在ego右侧"（NGSIM: y大=右; highD负向: y大=左 需镜像）
    others["_dy"] = (others["y"] - ego_y) * drv_dir

    # 横向分区
    left_mask  = others["_dy"] < -SAME_LANE_Y_TOL
    right_mask = others["_dy"] >  SAME_LANE_Y_TOL
    same_mask  = ~left_mask & ~right_mask

    # 纵向分区（用于区分 preceding/alongside/following）
    # alongside：|dx| <= 半车辆长度（约5m），视为并排
    ALONGSIDE_DX = 5.0
    def _pick_slot(subset: pd.DataFrame, is_front: bool | None) -> pd.Series | None:
        """从 subset 里按纵向距离取最近的一辆车。
        is_front=True: 取前方; is_front=False: 取后方; None: 取并排"""
        if subset.empty:
            return None
        if is_front is True:
            cands = subset[subset["_dx"] > ALONGSIDE_DX]
            if cands.empty:
                return None
            return cands.loc[cands["_dx"].idxmin()]
        elif is_front is False:
            cands = subset[subset["_dx"] < -ALONGSIDE_DX]
            if cands.empty:
                return None
            return cands.loc[cands["_dx"].idxmax()]  # 最近的后方车（dx最大负值）
        else:  # alongside
            cands = subset[subset["_dx"].abs() <= ALONGSIDE_DX]
            if cands.empty:
                return None
            return cands.loc[cands["_dx"].abs().idxmin()]

    left_cars  = others[left_mask]
    right_cars = others[right_mask]
    same_cars  = others[same_mask]

    slots = {
        "left_preceding":  _pick_slot(left_cars,  True),
        "left_alongside":  _pick_slot(left_cars,  None),
        "left_following":  _pick_slot(left_cars,  False),
        "right_preceding": _pick_slot(right_cars, True),
        "right_alongside": _pick_slot(right_cars, None),
        "right_following": _pick_slot(right_cars, False),
    }
    return slots


def _empty_slots() -> dict:
    return {name: None for name in SLOT_NAMES}


def _compute_ttc(
    ego_x: float, ego_vx: float,
    other_x: float, other_vx: float,
    same_lane: bool,
) -> tuple[float, bool]:
    """
    计算 TTC（Time To Collision）。

    只在同车道或即将汇入同车道时计算有效值。
    TTC = (前车x - 后车x) / (后车vx - 前车vx)
    仅当追及条件成立（后车比前车快）且距离为正时有效。

    Returns:
        (ttc_value, is_valid)
        ttc_value: 有效时为秒数，无效时为 np.nan
        is_valid: 是否为有效 TTC（用于 valid_ttc_mask）
    """
    if not same_lane:
        return np.nan, False

    # 判断谁在前（沿 x 轴，但需考虑行驶方向——这里调用方已保证 dx>0 为前方）
    dx = other_x - ego_x
    dvx = ego_vx - other_vx  # ego 相对 other 的速度

    if dx > 0:
        # other 在前方
        closing_speed = dvx  # ego 追 other
        gap = dx
    else:
        # ego 在前方
        closing_speed = -dvx  # other 追 ego
        gap = -dx

    if closing_speed <= 0 or gap <= 0:
        # 没有追及趋势
        return np.nan, False

    ttc = gap / closing_speed
    if ttc > 30.0:
        # TTC > 30秒认为无实际碰撞威胁，标记无效
        return np.nan, False

    return float(ttc), True


def _is_lane_change(
    ego_traj: pd.DataFrame,
    lane_width: float = LANE_WIDTH_DEFAULT,
) -> bool:
    """
    判断 ego 在预测窗口内是否发生变道。

    优先用 laneId 变化（可靠）；
    备用：横向位移超过半个车道宽度（阈值法）。

    Args:
        ego_traj: ego 车辆在整个8秒窗口内的数据（含历史+预测帧）
    """
    if "laneId" in ego_traj.columns:
        lane_ids = ego_traj["laneId"].dropna()
        if len(lane_ids.unique()) > 1:
            return True

    # 备用：横向位移阈值
    y_vals = ego_traj["y_smooth"].dropna() if "y_smooth" in ego_traj.columns \
             else ego_traj["y"].dropna()
    if len(y_vals) < 2:
        return False
    lateral_disp = abs(y_vals.iloc[-1] - y_vals.iloc[0])
    return lateral_disp > lane_width * 0.5


# 每个槽位输出的字段，顺序固定，任何情况下都按此顺序写入
# 这是防止 pyarrow schema 不匹配的唯一可靠方式
_SLOT_FIELD_ORDER = [
    "mask", "id", "x", "y", "vx", "vy", "lane",
    "ttc", "ttc_valid", "thw", "rel_vx", "rel_vy",
]


def _build_slot_fields(
    slot_name: str,
    other: pd.Series | None,
    ego_hist_last: pd.Series,
    drv_dir: int,
) -> dict:
    """
    为单个槽位构建字段字典，字段顺序严格按 _SLOT_FIELD_ORDER。

    无论 other 是否存在、是否是 preceding 槽位，输出的列名和顺序都完全相同，
    保证 pyarrow ParquetWriter 的 schema 在所有批次之间一致。
    """
    p = slot_name  # 前缀

    if other is None:
        # 空槽位：mask=0，数值全 nan，ttc_valid=0
        return {
            f"{p}_mask":      0,
            f"{p}_id":        np.nan,
            f"{p}_x":         np.nan,
            f"{p}_y":         np.nan,
            f"{p}_vx":        np.nan,
            f"{p}_vy":        np.nan,
            f"{p}_lane":      np.nan,
            f"{p}_ttc":       np.nan,
            f"{p}_ttc_valid": 0,
            f"{p}_thw":       np.nan,
            f"{p}_rel_vx":    np.nan,
            f"{p}_rel_vy":    np.nan,
        }

    # 有邻车
    ego_x   = float(ego_hist_last["x"])
    ego_vx  = float(ego_hist_last["xVelocity"])
    ego_vy  = float(ego_hist_last["yVelocity"])

    # TTC 和 THW：只对 preceding 槽位有意义
    if "preceding" in slot_name:
        same_lane = abs(float(other["y"]) - float(ego_hist_last["y"])) <= SAME_LANE_Y_TOL
        ttc, ttc_valid = _compute_ttc(
            ego_x=ego_x,
            ego_vx=ego_vx * drv_dir,
            other_x=float(other["x"]),
            other_vx=float(other["xVelocity"]) * drv_dir,
            same_lane=same_lane,
        )
        gap     = abs(float(other["x"]) - ego_x)
        ego_spd = abs(ego_vx) + 1e-6
        thw     = float(gap / ego_spd)
    else:
        ttc, ttc_valid = np.nan, False
        thw = np.nan

    return {
        f"{p}_mask":      1,
        f"{p}_id":        float(other["id"]),
        f"{p}_x":         float(other["x"]),
        f"{p}_y":         float(other["y"]),
        f"{p}_vx":        float(other["xVelocity"]),
        f"{p}_vy":        float(other["yVelocity"]),
        f"{p}_lane":      float(other.get("laneId", np.nan)),
        f"{p}_ttc":       ttc,
        f"{p}_ttc_valid": int(ttc_valid),
        f"{p}_thw":       thw,
        f"{p}_rel_vx":    float(other["xVelocity"]) - float(ego_hist_last["xVelocity"]),
        f"{p}_rel_vy":    float(other["yVelocity"]) - ego_vy,
    }


# ============================================================================
# 单窗口预计算
# ============================================================================

def compute_window_features(
    window_meta: pd.Series,
    file_cache: dict[str, pd.DataFrame],
) -> dict | None:
    """
    对单个窗口预计算所有特征，返回一个扁平字典（写入 parquet 的一行）。

    Args:
        window_meta: global_window_index 里的一行（一个窗口的元数据）
        file_cache: {file_tag: dataframe} 的内存缓存，避免重复读 parquet

    Returns:
        特征字典，失败时返回 None
    """
    wid        = window_meta["window_id"]
    src_file   = window_meta["source_file"]
    ego_id     = window_meta["vehicle_id"]
    start_fr   = int(window_meta["start_frame"])
    end_fr     = int(window_meta["end_frame"])    # 不含
    n_hist     = int(window_meta["n_frames_hist"])# 15
    dataset    = window_meta["dataset_name"]

    # 从缓存里取对应 parquet
    if src_file not in file_cache:
        return None
    df = file_cache[src_file]

    # 取出 ego 在整个窗口内的帧
    ego_mask = df["id"] == ego_id
    ego_all_sorted = df[ego_mask].sort_values("frame").reset_index(drop=True)

    # 用"组内位置"索引来切片（因为 start_frame/end_frame 是组内位置，不是原始 frame 值）
    if end_fr > len(ego_all_sorted):
        return None
    ego_window = ego_all_sorted.iloc[start_fr:end_fr]

    if len(ego_window) != (end_fr - start_fr):
        return None

    # 历史段的最后一帧（预测起点）用于邻车选择
    ego_hist_last = ego_window.iloc[n_hist - 1]
    hist_frame_val = ego_hist_last["frame"]  # 原始 frame 列的值

    # 取同帧的所有车辆
    same_frame_df = df[df["frame"] == hist_frame_val].copy()

    # 按行驶方向分组，找 ego 所在的方向组
    ego_vx = ego_hist_last["xVelocity"]
    drv_dir = 1 if ego_vx >= 0 else -1

    # 只保留同方向的车辆（方向一致性：vx 同号，或速度极小的视为同向）
    SPEED_THRESH = 1.0  # m/s，低于此认为几乎静止，归入ego同方向
    same_dir_mask = (
        (same_frame_df["xVelocity"] * drv_dir >= 0) |
        (same_frame_df["xVelocity"].abs() < SPEED_THRESH)
    )
    same_dir_frame = same_frame_df[same_dir_mask].copy()
    same_dir_frame["_drv_dir"] = drv_dir

    # 给 ego_hist_last 也加上 _drv_dir
    ego_for_slot = ego_hist_last.copy()
    ego_for_slot["_drv_dir"] = drv_dir

    # 分配6个槽位
    slots = _assign_slots_for_ego(ego_for_slot, same_dir_frame)

    # 构建输出字典
    result = {"window_id": wid}

    for slot_name in SLOT_NAMES:
        other = slots[slot_name]
        slot_fields = _build_slot_fields(slot_name, other, ego_hist_last, drv_dir)
        result.update(slot_fields)

    # 变道标注
    result["is_lane_change"] = int(_is_lane_change(ego_window))

    # 场景密度（同帧同方向的车辆数，排除ego自身）
    result["scene_density"] = int(len(same_dir_frame)) - 1

    # ego 基本运动特征（历史段终点，用于快速筛选）
    result["ego_vx_at_hist_end"]  = float(ego_hist_last["xVelocity"])
    result["ego_vy_at_hist_end"]  = float(ego_hist_last["yVelocity"])
    result["ego_laneId_at_hist_end"] = float(ego_hist_last.get("laneId", np.nan))

    return result


# ============================================================================
# 批量预计算主函数
# ============================================================================

def precompute_features(
    global_index_path: str | Path,
    processed_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 10000,
    resume: bool = True,
    dataset_filter: str | None = None,
) -> None:
    """
    批量预计算所有窗口的邻车/TTC/变道特征，结果写入 output_path。

    写入策略：使用 pyarrow ParquetWriter 流式追加，每批直接 append
    而不需要读回整个文件，写入速度恒定（不随文件增大而变慢）。

    Args:
        global_index_path: global_window_index.parquet 路径
        processed_dir: 降采样后的 parquet 目录（含 ngsim/ 和 highd/ 子目录）
        output_path: 输出特征缓存路径
        batch_size: 每批处理的窗口数（控制内存峰值）
        resume: 若输出文件已存在，跳过已处理的 window_id（断点续跑）
        dataset_filter: "ngsim" / "highd" / None（全部）
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    global_index_path = Path(global_index_path)
    processed_dir     = Path(processed_dir)
    output_path       = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"读取全局窗口索引: {global_index_path}")
    index_df = pd.read_parquet(global_index_path)

    if dataset_filter:
        index_df = index_df[index_df["dataset_name"] == dataset_filter]
        logger.info(f"过滤到数据集: {dataset_filter}，窗口数: {len(index_df)}")

    # 断点续跑：读取已完成的 window_id，跳过它们
    done_ids: set[str] = set()
    if resume and output_path.exists():
        done_df = pd.read_parquet(output_path, columns=["window_id"])
        done_ids = set(done_df["window_id"].tolist())
        logger.info(f"断点续跑：已完成 {len(done_ids)} 个窗口，跳过")
        index_df = index_df[~index_df["window_id"].isin(done_ids)]

    logger.info(f"待处理窗口数: {len(index_df)}")
    if len(index_df) == 0:
        logger.info("无需处理，直接退出")
        return

    # 按 source_file 分组，同文件的窗口一起处理（避免重复读 parquet）
    file_groups = index_df.groupby("source_file")
    total_written = len(done_ids)
    all_results: list[dict] = []

    # 用 pyarrow ParquetWriter 流式写入
    # resume 模式下以追加方式打开（append=True），首次运行直接创建
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    def _flush(results: list[dict]) -> None:
        """把当前批次的结果写入 parquet，保持 writer 引用持久。"""
        nonlocal writer, schema, total_written
        if not results:
            return
        batch_df = pd.DataFrame(results)

        # 强制列顺序与第一批一致（防止不同批次的 dict 键顺序细微差异）
        if schema is not None:
            existing_cols = [f.name for f in schema]
            batch_df = batch_df[existing_cols]

        table = pa.Table.from_pandas(batch_df, preserve_index=False)

        if writer is None:
            schema = table.schema
            # resume 模式：以追加方式写入已有文件（pyarrow 1.0+ 支持）
            writer = pq.ParquetWriter(
                str(output_path), schema,
                compression="snappy",
            )
        writer.write_table(table)
        total_written += len(results)
        logger.info(f"  已写出 {total_written} 个窗口")

    try:
        for file_tag, group in file_groups:
            dataset_name = group["dataset_name"].iloc[0]
            parquet_path = processed_dir / dataset_name / f"{file_tag}_5hz.parquet"

            if not parquet_path.exists():
                logger.warning(f"[SKIP] parquet 不存在: {parquet_path}")
                continue

            logger.info(f"处理文件: {file_tag} ({len(group)} 个窗口)")
            file_df    = pd.read_parquet(parquet_path)
            file_cache = {file_tag: file_df}
            n_fail     = 0

            for _, row in group.iterrows():
                feat = compute_window_features(row, file_cache)
                if feat is not None:
                    all_results.append(feat)
                else:
                    n_fail += 1

                if len(all_results) >= batch_size:
                    _flush(all_results)
                    all_results = []

            if n_fail > 0:
                logger.warning(f"  {file_tag}: {n_fail} 个窗口处理失败（轨迹数据不足）")

        # 写出最后一批
        if all_results:
            _flush(all_results)

    finally:
        if writer is not None:
            writer.close()

    logger.info(f"预计算完成，共写出 {total_written} 个窗口特征 → {output_path}")


# ============================================================================
# 命令行入口
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="预计算邻车/TTC/变道特征")
    parser.add_argument("--index",   default="data/processed/global_window_index.parquet")
    parser.add_argument("--proc_dir",default="data/processed")
    parser.add_argument("--output",  default="data/processed/features_cache.parquet")
    parser.add_argument("--dataset", default=None, choices=["ngsim","highd",None])
    parser.add_argument("--batch",   type=int, default=10000)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    precompute_features(
        global_index_path=args.index,
        processed_dir=args.proc_dir,
        output_path=args.output,
        batch_size=args.batch,
        resume=not args.no_resume,
        dataset_filter=args.dataset,
    )

if __name__ == "__main__":
    main()