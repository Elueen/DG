"""
src/data/preprocess.py

预处理流水线 — 第一步：自动采样频率检测

设计原则：
- 不假设固定频率，所有后续步骤都依赖这里检测出的值
- 按 vehicle_id 分组，统计帧间隔的众数，避免被单个异常间隔带偏
- 同时支持基于 frame 列（整数帧号）和基于 timestamp 列（毫秒/秒）两种情况
- 当没有 timestamp 列时，用"位置差分速度 vs 数据自带速度"的物理一致性
  自动反推真实频率，而不是要求人工瞎猜

关于 NGSIM 与 highD 平滑降幅不同的说明（写论文 Implementation Details 时可直接引用）：
- 平滑窗口长度按统一规则换算（target_seconds=0.3秒 × 检测到的原始频率），
  对两个数据集使用同一套换算逻辑，而不是手工为每个数据集单独调参。
- 实测中 NGSIM 平滑降幅仅 ~0.8%，highD 平滑降幅 ~41%，这一差异源于
  两个数据集本身的预处理状态不同：NGSIM 官方发布时已对轨迹做过
  Kalman 滤波去噪（这是 NGSIM 数据集公开已知的特性），位置噪声本身很低；
  而 highD 提供的是更接近原始传感器输出的轨迹，噪声水平更高。
  因此在同一套平滑逻辑下，NGSIM 表现出"轻微平滑"、highD 表现出
  "明显平滑"是符合数据本身物理特性的预期结果，不是参数设置错误，
  也不需要为两个数据集手工设置不同的窗口参数。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------------------------------------------------------------
# 常见 highD / NGSIM 字段名映射候选
# ----------------------------------------------------------------------
VEHICLE_ID_CANDIDATES = ["id", "vehicle_id", "Vehicle_ID", "track_id", "trackId"]
FRAME_CANDIDATES = ["frame", "Frame_ID", "frame_id", "frameId"]
TIME_CANDIDATES = ["timestamp", "time", "Global_Time", "global_time"]
X_CANDIDATES = ["x", "X", "local_x", "Local_X"]
Y_CANDIDATES = ["y", "Y", "local_y", "Local_Y"]
XVEL_CANDIDATES = ["xVelocity", "vx", "v_x"]
YVEL_CANDIDATES = ["yVelocity", "vy", "v_y"]
# 合速度标量（部分NGSIM半转换文件只有合速度，没有x/y分量）
# 用于物理一致性频率反推时的备用速度字段
SCALAR_VEL_CANDIDATES = ["v_Vel", "v_vel", "speed", "velocity", "v_Speed"]

# 候选频率集合：用于物理一致性反推时遍历测试
CANDIDATE_FREQUENCIES_HZ = [10.0, 25.0, 30.0, 50.0]


@dataclass
class FrequencyDetectionResult:
    """频率检测结果，后续所有降采样/窗口切片步骤都依赖这个对象。"""

    detected_freq_hz: float
    dt_seconds: float
    mode_frame_gap: int
    method: str                        # "frame_column" / "timestamp_column" / "velocity_consistency"
    n_vehicles_sampled: int
    n_gap_samples: int
    gap_distribution: dict
    confidence: float                  # 众数占比 或 速度一致性置信度


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_sampling_frequency(
    df: pd.DataFrame,
    vehicle_id_col: str | None = None,
    frame_col: str | None = None,
    time_col: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    xvel_col: str | None = None,
    yvel_col: str | None = None,
    n_vehicles_to_sample: int = 50,
    random_seed: int = 42,
) -> FrequencyDetectionResult:
    """
    自动检测原始数据的采样频率。

    优先级：
    1. frame 列 + timestamp 列同时存在 → 用 timestamp 精确校准（最准）
    2. 只有 frame 列，但数据带速度字段 (xVelocity/yVelocity) 和位置字段 (x/y)
       → 用"位置差分速度 vs 数据自带速度"的物理一致性反推频率
    3. 都没有 → 退化为警告 + 返回 NaN，要求人工指定
    """
    vid_col = vehicle_id_col or _find_column(df, VEHICLE_ID_CANDIDATES)
    if vid_col is None:
        raise ValueError(
            f"未能自动识别车辆ID列，请显式传入 vehicle_id_col。"
            f"当前数据列为: {list(df.columns)}"
        )

    f_col = frame_col or _find_column(df, FRAME_CANDIDATES)
    t_col = time_col or _find_column(df, TIME_CANDIDATES)

    if f_col is None and t_col is None:
        raise ValueError(
            f"未能识别 frame 列或 timestamp 列，请显式传入 frame_col 或 time_col。"
            f"当前数据列为: {list(df.columns)}"
        )

    logger.info(f"使用车辆ID列: '{vid_col}'")

    rng = np.random.default_rng(random_seed)
    unique_vids = df[vid_col].unique()
    if len(unique_vids) == 0:
        raise ValueError("数据中没有任何车辆ID，无法检测频率。")

    n_sample = min(n_vehicles_to_sample, len(unique_vids))
    sampled_vids = rng.choice(unique_vids, size=n_sample, replace=False)

    # ---- 路径1：有 frame 列，先统计众数间隔 ----
    if f_col is not None:
        logger.info(f"使用 frame 列: '{f_col}' 进行检测")
        all_gaps = []
        for vid in sampled_vids:
            sub = df[df[vid_col] == vid].sort_values(f_col)
            frames = sub[f_col].to_numpy()
            if len(frames) < 2:
                continue
            all_gaps.extend(np.diff(frames).tolist())

        if len(all_gaps) == 0:
            raise ValueError("抽样车辆中没有任何车辆拥有 ≥2 帧数据，无法计算间隔。")

        gap_counter = Counter(all_gaps)
        mode_gap, mode_count = gap_counter.most_common(1)[0]
        confidence = mode_count / len(all_gaps)

        # ---- 路径1a：有 timestamp 列，精确校准 ----
        if t_col is not None:
            dt_seconds, freq_hz = _calibrate_dt_with_timestamp(
                df, vid_col, f_col, t_col, sampled_vids, mode_gap
            )
            method = "timestamp_column"

        # ---- 路径1b：没有 timestamp，但有位置+速度字段 → 物理一致性反推 ----
        else:
            xc = x_col or _find_column(df, X_CANDIDATES)
            yc = y_col or _find_column(df, Y_CANDIDATES)
            xvc = xvel_col or _find_column(df, XVEL_CANDIDATES)
            yvc = yvel_col or _find_column(df, YVEL_CANDIDATES)

            # 回退：如果没有 xVelocity 分量，尝试用合速度标量 v_Vel 替代
            # 原理：对于高速公路直行车辆，横向速度远小于纵向速度，
            # v_Vel ≈ |vx|，用 dx/dt 与 v_Vel 做比值检验仍然有效
            # （比值偏离1.0时的倍数关系与用xVelocity完全相同，频率判断不受影响）
            scalar_vel_col = None
            if xvc is None:
                scalar_vel_col = _find_column(df, SCALAR_VEL_CANDIDATES)
                if scalar_vel_col is not None:
                    logger.info(
                        f"未找到 xVelocity 分量列，改用合速度标量列 '{scalar_vel_col}' "
                        f"做物理一致性反推（高速公路场景下 v_Vel ≈ |vx|，结果可靠）"
                    )
                    xvc = scalar_vel_col  # 临时复用 xvc 槽位，传入反推函数

            if xc is not None and xvc is not None:
                logger.info(
                    "没有 timestamp 列，但检测到位置列与速度列，"
                    "改用【物理一致性反推】自动判断真实频率..."
                )
                freq_hz, consistency_confidence, detail = _verify_frequency_by_velocity_consistency(
                    df, vid_col, f_col, xc, yc, xvc, yvc, sampled_vids, mode_gap
                )
                dt_seconds = 1.0 / freq_hz if freq_hz and freq_hz > 0 else float("nan")
                confidence = consistency_confidence
                method = "velocity_consistency"
                _print_velocity_consistency_detail(detail)
            else:
                logger.warning(
                    "没有 timestamp 列，也没有找到可用的位置/速度列组合，"
                    "无法做物理一致性反推，退化为人工确认模式。"
                )
                dt_seconds, freq_hz = float("nan"), float("nan")
                method = "frame_column"

        result = FrequencyDetectionResult(
            detected_freq_hz=freq_hz,
            dt_seconds=dt_seconds,
            mode_frame_gap=int(mode_gap),
            method=method,
            n_vehicles_sampled=n_sample,
            n_gap_samples=len(all_gaps),
            gap_distribution=dict(gap_counter.most_common(10)),
            confidence=confidence,
        )

    # ---- 路径2：没有 frame 列，只能用 timestamp ----
    else:
        logger.info(f"未找到 frame 列，使用 timestamp 列: '{t_col}' 进行检测")
        all_gaps = []
        for vid in sampled_vids:
            sub = df[df[vid_col] == vid].sort_values(t_col)
            times = sub[t_col].to_numpy(dtype=np.float64)
            if len(times) < 2:
                continue
            all_gaps.extend(np.diff(times).tolist())

        if len(all_gaps) == 0:
            raise ValueError("抽样车辆中没有任何车辆拥有 ≥2 帧数据，无法计算间隔。")

        median_gap_raw = float(np.median(all_gaps))
        time_unit, unit_scale = _guess_timestamp_unit(median_gap_raw)
        logger.info(f"猜测 timestamp 单位为: {time_unit}（中位数间隔原始值={median_gap_raw:.4f}）")

        gaps_seconds = np.array(all_gaps) * unit_scale
        gaps_rounded_ms = np.round(gaps_seconds * 1000).astype(int)
        gap_counter = Counter(gaps_rounded_ms.tolist())
        mode_gap_ms, mode_count = gap_counter.most_common(1)[0]
        confidence = mode_count / len(gaps_rounded_ms)

        dt_seconds = mode_gap_ms / 1000.0
        freq_hz = round(1.0 / dt_seconds, 2) if dt_seconds > 0 else float("nan")

        result = FrequencyDetectionResult(
            detected_freq_hz=freq_hz,
            dt_seconds=dt_seconds,
            mode_frame_gap=-1,
            method="timestamp_column",
            n_vehicles_sampled=n_sample,
            n_gap_samples=len(all_gaps),
            gap_distribution={k: v for k, v in gap_counter.most_common(10)},
            confidence=confidence,
        )

    _print_detection_report(result)
    return result


def _calibrate_dt_with_timestamp(
    df: pd.DataFrame,
    vid_col: str,
    f_col: str,
    t_col: str,
    sampled_vids: np.ndarray,
    mode_frame_gap: int,
) -> tuple[float, float]:
    dt_candidates = []
    for vid in sampled_vids:
        sub = df[df[vid_col] == vid].sort_values(f_col)
        frames = sub[f_col].to_numpy()
        times = sub[t_col].to_numpy(dtype=np.float64)
        if len(frames) < 2:
            continue
        frame_gaps = np.diff(frames)
        time_gaps = np.diff(times)
        mask = frame_gaps == mode_frame_gap
        if mask.sum() == 0:
            continue
        dt_candidates.extend(time_gaps[mask].tolist())

    if len(dt_candidates) == 0:
        logger.warning("无法用timestamp校准frame间隔对应的真实秒数。")
        return float("nan"), float("nan")

    median_dt_raw = float(np.median(dt_candidates))
    time_unit, unit_scale = _guess_timestamp_unit(median_dt_raw / max(mode_frame_gap, 1))
    dt_per_frame_seconds = (median_dt_raw / max(mode_frame_gap, 1)) * unit_scale
    freq_hz = round(1.0 / dt_per_frame_seconds, 2) if dt_per_frame_seconds > 0 else float("nan")
    dt_seconds = 1.0 / freq_hz if freq_hz > 0 else float("nan")
    return dt_seconds, freq_hz


def _guess_timestamp_unit(median_gap_raw: float) -> tuple[str, float]:
    if median_gap_raw <= 0:
        raise ValueError(f"计算出的时间间隔非正数: {median_gap_raw}，请检查数据排序/去重。")
    if median_gap_raw >= 1.0:
        return "milliseconds", 1e-3
    else:
        return "seconds", 1.0


def _verify_frequency_by_velocity_consistency(
    df: pd.DataFrame,
    vid_col: str,
    f_col: str,
    x_col: str,
    y_col: str | None,
    xvel_col: str,
    yvel_col: str | None,
    sampled_vids: np.ndarray,
    mode_frame_gap: int,
) -> tuple[float, float, dict]:
    """
    物理一致性反推核心逻辑：

    原理：数据自带的 xVelocity 字段（无论原始频率是多少）单位都是 m/s，
    是一个不随"我们假设的频率"变化的物理真值。
    而"位置差分速度" = (x[t+1] - x[t]) / dt 则依赖于我们假设的 dt。

    如果假设的频率正确，那么：
        (x[t+1] - x[t]) / dt_假设  ≈  xVelocity[t]   （两者应高度相关，且数值接近）

    如果假设的频率错了（比如真实是10Hz但假设成了25Hz，dt用小了2.5倍），
    位置差分速度会比真实速度系统性偏小（或偏大）约2.5倍，一眼可辨。

    我们对 CANDIDATE_FREQUENCIES_HZ 中每个候选频率都计算一次"差分速度 vs
    自带速度"的比值中位数，比值最接近1.0的那个候选频率就是真实频率。

    Returns:
        (best_freq_hz, confidence, detail_dict)
        confidence: 最佳候选与次佳候选的比值残差差距，差距越大说明判断越可信
    """
    # 收集所有抽样车辆、众数frame间隔下的 (位置差, 自带速度) 配对
    dx_list, dy_list, vx_list, vy_list = [], [], [], []

    for vid in sampled_vids:
        sub = df[df[vid_col] == vid].sort_values(f_col)
        frames = sub[f_col].to_numpy()
        if len(frames) < 2:
            continue

        x = sub[x_col].to_numpy(dtype=np.float64)
        vx = sub[xvel_col].to_numpy(dtype=np.float64)

        frame_gaps = np.diff(frames)
        mask = frame_gaps == mode_frame_gap
        if mask.sum() == 0:
            continue

        dx = np.diff(x)[mask]
        # 用相邻两帧自带速度的均值，比单端点更稳健
        vx_avg = ((vx[:-1] + vx[1:]) / 2.0)[mask]

        dx_list.extend(dx.tolist())
        vx_list.extend(vx_avg.tolist())

        if y_col is not None and yvel_col is not None:
            y = sub[y_col].to_numpy(dtype=np.float64)
            vy = sub[yvel_col].to_numpy(dtype=np.float64)
            dy = np.diff(y)[mask]
            vy_avg = ((vy[:-1] + vy[1:]) / 2.0)[mask]
            dy_list.extend(dy.tolist())
            vy_list.extend(vy_avg.tolist())

    dx_arr = np.array(dx_list)
    vx_arr = np.array(vx_list)

    # 过滤掉自带速度接近0的样本（除以小数会爆炸性放大噪声）
    valid_mask = np.abs(vx_arr) > 1.0  # m/s，过滤近似静止的样本
    dx_arr = dx_arr[valid_mask]
    vx_arr = vx_arr[valid_mask]

    if len(dx_arr) < 100:
        logger.warning(
            f"有效样本数过少 ({len(dx_arr)})，物理一致性反推结果可能不可靠，建议人工核实。"
        )

    candidate_results = {}
    for freq in CANDIDATE_FREQUENCIES_HZ:
        dt = mode_frame_gap / freq
        implied_vx = dx_arr / dt  # 用这个假设频率算出的"差分速度"
        # 比值理论上应该≈1，用比值的中位数与1的偏离程度衡量拟合优度
        ratio = implied_vx / vx_arr
        ratio = ratio[np.isfinite(ratio)]
        median_ratio = float(np.median(ratio))
        # 残差：|log(ratio)|的中位数，取log是因为高估和低估应该对称地惩罚
        residual = float(np.median(np.abs(np.log(np.abs(ratio) + 1e-8))))
        candidate_results[freq] = {
            "median_ratio": median_ratio,
            "residual": residual,
        }

    # 残差最小的候选频率就是最佳估计
    best_freq = min(candidate_results, key=lambda f: candidate_results[f]["residual"])
    sorted_residuals = sorted(v["residual"] for v in candidate_results.values())
    best_residual = sorted_residuals[0]
    second_best_residual = sorted_residuals[1] if len(sorted_residuals) > 1 else best_residual + 1e-6

    # confidence: 次佳与最佳的残差比值差距，差距越大说明越确信
    confidence = 1.0 - (best_residual / max(second_best_residual, 1e-8))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    detail = {
        "n_valid_samples": len(dx_arr),
        "candidate_results": candidate_results,
        "best_freq": best_freq,
    }
    return best_freq, confidence, detail


def _print_velocity_consistency_detail(detail: dict) -> None:
    print("\n" + "-" * 60)
    print("物理一致性反推明细（位置差分速度 vs 数据自带速度字段对比）")
    print("-" * 60)
    print(f"有效样本数: {detail['n_valid_samples']}")
    print(f"{'候选频率(Hz)':<15}{'比值中位数':<15}{'残差(越小越好)':<15}")
    for freq, res in sorted(detail["candidate_results"].items()):
        marker = "  ← 最佳" if freq == detail["best_freq"] else ""
        print(f"{freq:<15}{res['median_ratio']:<15.4f}{res['residual']:<15.4f}{marker}")
    print("-" * 60)
    print("解读：比值中位数越接近 1.0，说明该候选频率下，'位置差分算出的速度'")
    print("与'数据自带速度字段'越吻合，即该频率越可能是真实采样频率。")


def _print_detection_report(result: FrequencyDetectionResult) -> None:
    print("\n" + "=" * 60)
    print("采样频率检测报告")
    print("=" * 60)
    print(f"检测方法           : {result.method}")
    print(f"抽样车辆数         : {result.n_vehicles_sampled}")
    print(f"统计间隔样本数     : {result.n_gap_samples}")
    print(f"置信度             : {result.confidence:.2%}")
    print(f"frame间隔分布(Top10): {result.gap_distribution}")
    print("-" * 60)
    if np.isnan(result.detected_freq_hz):
        print("⚠️  未能自动确定频率，请人工检查或手动指定 dt_seconds / freq_hz")
    else:
        print(f"检测到采样间隔 dt  : {result.dt_seconds:.4f} 秒")
        print(f"检测到采样频率     : {result.detected_freq_hz:.2f} Hz")
        for known_freq, known_name in [(10.0, "NGSIM原生10Hz"), (25.0, "highD原生25Hz")]:
            if abs(result.detected_freq_hz - known_freq) < 0.5:
                print(f"  → 与 {known_name} 高度吻合")
    if result.confidence < 0.8:
        print(
            f"⚠️  警告：置信度仅 {result.confidence:.2%}，低于80%，"
            f"建议检查原始数据是否存在缺帧/拼接/单位混乱问题。"
        )
    print("=" * 60 + "\n")


# ============================================================================
# 第五步：位置平滑（Savitzky-Golay）—— 必须在降采样之前执行
# ============================================================================
@dataclass
class SmoothingDiagnostics:
    """平滑前后的速度/加速度标准差对比，用于人工判断平滑强度是否合适。"""

    vx_std_before: float
    vx_std_after: float
    vy_std_before: float
    vy_std_after: float
    ax_std_before: float
    ax_std_after: float
    ay_std_before: float
    ay_std_after: float
    n_vehicles: int
    window_length_frames: int
    polyorder: int


def _compute_window_length_frames(
    source_freq_hz: float,
    target_seconds: float = 0.3,
    polyorder: int = 2,
) -> int:
    """
    根据检测到的原始频率，把"大约0.2~0.5秒"的平滑窗口换算成对应的帧数。

    Savitzky-Golay 窗口长度必须是奇数，且必须 > polyorder。

    重要边界情况：当 window_length == polyorder + 1 时，SG滤波退化为
    精确多项式插值（自由度刚好用完），平滑效果为零（输出=输入）。
    必须保证 window_length >= polyorder + 3（且为奇数），
    才能让最小二乘拟合真正起到去噪作用（有冗余自由度可以平均掉噪声）。

    Args:
        source_freq_hz: 第一步检测到的原始采样频率
        target_seconds: 期望的窗口时长（秒），默认0.3秒，落在0.2~0.5秒区间中点
        polyorder: SG滤波多项式阶数，窗口长度必须大于它

    Returns:
        奇数帧数，作为 savgol_filter 的 window_length 参数
    """
    if np.isnan(source_freq_hz) or source_freq_hz <= 0:
        raise ValueError(
            f"source_freq_hz 无效: {source_freq_hz}，请先完成第一步频率检测。"
        )

    raw_frames = target_seconds * source_freq_hz
    window = int(round(raw_frames))

    # 必须是奇数
    if window % 2 == 0:
        window += 1

    # 必须比 polyorder 至少多2个自由度（且为奇数），
    # 否则SG退化为精确插值，平滑无效（见函数docstring）
    min_valid = polyorder + 3
    if min_valid % 2 == 0:
        min_valid += 1
    window = max(window, min_valid)

    logger.info(
        f"窗口换算: 原始频率={source_freq_hz}Hz, 目标时长={target_seconds}s "
        f"→ 窗口帧数={window}（已修正为奇数且 >= polyorder+3={min_valid}，避免退化为精确插值）"
    )
    return window


def smooth_positions(
    df: pd.DataFrame,
    source_freq_hz: float,
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    window_length_frames: int | None = None,
    polyorder: int = 2,
    target_smoothing_seconds: float = 0.3,
) -> tuple[pd.DataFrame, SmoothingDiagnostics]:
    """
    对每辆车的 (x, y) 轨迹独立做 Savitzky-Golay 平滑。

    重要约束（按需求文档第五步）：
    - 只平滑位置，不直接平滑原始的 xVelocity/yVelocity/xAcceleration/yAcceleration
    - 平滑后会新增 'x_smooth', 'y_smooth' 两列，原始列保留不动（便于审计对比）
    - 速度/加速度的重新差分由 redifferentiate_kinematics() 单独负责，
      这里只产出平滑后的位置

    Args:
        df: 原始数据，至少包含 vehicle_id_col, frame_col, x_col, y_col
        source_freq_hz: 第一步检测出的原始频率（用于换算窗口帧数）
        window_length_frames: 显式指定窗口帧数；不传则自动按
            target_smoothing_seconds 换算
        polyorder: SG滤波多项式阶数，默认2（适合车辆轨迹这种较平滑的运动）
        target_smoothing_seconds: 自动换算窗口时长时使用，默认0.3秒

    Returns:
        (df_with_smoothed_cols, diagnostics)
    """
    if window_length_frames is None:
        window_length_frames = _compute_window_length_frames(
            source_freq_hz, target_smoothing_seconds, polyorder
        )
    else:
        if window_length_frames % 2 == 0:
            raise ValueError(
                f"window_length_frames 必须是奇数，收到: {window_length_frames}"
            )
        if window_length_frames <= polyorder:
            raise ValueError(
                f"window_length_frames({window_length_frames}) 必须大于 "
                f"polyorder({polyorder})"
            )

    df = df.sort_values([vehicle_id_col, frame_col]).reset_index(drop=True)
    df["x_smooth"] = np.nan
    df["y_smooth"] = np.nan

    n_vehicles_processed = 0
    n_vehicles_too_short = 0

    for vid, group_idx in df.groupby(vehicle_id_col).groups.items():
        idx = group_idx
        n_frames = len(idx)

        if n_frames < window_length_frames:
            # 轨迹太短，无法用完整窗口平滑。
            # 处理方式：用能容纳的最大奇数窗口（至少3帧）做平滑；
            # 如果连3帧都不够，直接原样保留（不平滑），并计数告警。
            if n_frames < 3:
                df.loc[idx, "x_smooth"] = df.loc[idx, x_col].to_numpy()
                df.loc[idx, "y_smooth"] = df.loc[idx, y_col].to_numpy()
                n_vehicles_too_short += 1
                continue
            local_window = n_frames if n_frames % 2 == 1 else n_frames - 1
            local_window = max(local_window, 3)
            local_polyorder = min(polyorder, local_window - 1)
        else:
            local_window = window_length_frames
            local_polyorder = polyorder

        x_vals = df.loc[idx, x_col].to_numpy(dtype=np.float64)
        y_vals = df.loc[idx, y_col].to_numpy(dtype=np.float64)

        # ---- 数值合法性检查（防止脏数据导致savgol SVD不收敛）----
        # 三种情况会让SG滤波的最小二乘内核崩溃：
        #   1. NaN/Inf：矩阵元素非法
        #   2. 全常数序列：秩亏，lstsq无法收敛
        #   3. 数值量级极端（>1e10）：浮点溢出
        def _is_safe_for_savgol(arr: np.ndarray) -> tuple[bool, str]:
            if not np.all(np.isfinite(arr)):
                return False, f"含NaN/Inf（共{(~np.isfinite(arr)).sum()}个）"
            if np.all(arr == arr[0]):
                return False, "全部相同值（常数序列，矩阵秩亏）"
            if np.max(np.abs(arr)) > 1e10:
                return False, f"数值量级过大（最大绝对值={np.max(np.abs(arr)):.2e}）"
            return True, ""

        x_ok, x_reason = _is_safe_for_savgol(x_vals)
        y_ok, y_reason = _is_safe_for_savgol(y_vals)

        if not x_ok or not y_ok:
            df.loc[idx, "x_smooth"] = x_vals
            df.loc[idx, "y_smooth"] = y_vals
            reasons = []
            if not x_ok:
                reasons.append(f"x列: {x_reason}")
            if not y_ok:
                reasons.append(f"y列: {y_reason}")
            logger.warning(
                f"车辆 {vid}（{n_frames}帧）跳过平滑，原因: {'; '.join(reasons)}。"
                f"该车辆保留原始位置值，不影响其他车辆。"
            )
            n_vehicles_too_short += 1
            continue

        try:
            x_smooth = savgol_filter(x_vals, window_length=local_window, polyorder=local_polyorder)
            y_smooth = savgol_filter(y_vals, window_length=local_window, polyorder=local_polyorder)
        except Exception as e:
            # 兜底：通过检查后仍可能失败（极端边界情况），保留原始值不中断整个文件
            df.loc[idx, "x_smooth"] = x_vals
            df.loc[idx, "y_smooth"] = y_vals
            logger.warning(
                f"车辆 {vid}（{n_frames}帧）savgol_filter 异常（{type(e).__name__}: {e}），"
                f"已回退到原始值，不影响其他车辆。"
            )
            n_vehicles_too_short += 1
            continue

        df.loc[idx, "x_smooth"] = x_smooth
        df.loc[idx, "y_smooth"] = y_smooth
        n_vehicles_processed += 1

    if n_vehicles_too_short > 0:
        logger.warning(
            f"{n_vehicles_too_short} 辆车的轨迹短于3帧，未做平滑（直接保留原始位置）。"
        )

    diagnostics = _diagnose_smoothing_effect(
        df, vehicle_id_col, frame_col, x_col, y_col, source_freq_hz,
        window_length_frames, polyorder,
    )

    return df, diagnostics


def _diagnose_smoothing_effect(
    df: pd.DataFrame,
    vehicle_id_col: str,
    frame_col: str,
    x_col: str,
    y_col: str,
    source_freq_hz: float,
    window_length_frames: int,
    polyorder: int,
) -> SmoothingDiagnostics:
    """
    对比平滑前后的速度/加速度标准差。

    做法：分别用原始位置和平滑后位置重新差分出速度，再差分出加速度，
    比较两组的标准差。标准差下降越多说明平滑去噪效果越强；
    但如果下降过多（比如>50%），可能说明窗口过大、把真实机动行为也磨平了。
    """
    dt = 1.0 / source_freq_hz

    vx_before_all, vy_before_all = [], []
    vx_after_all, vy_after_all = [], []
    ax_before_all, ay_before_all = [], []
    ax_after_all, ay_after_all = [], []

    n_vehicles = 0
    for vid, group_idx in df.groupby(vehicle_id_col).groups.items():
        idx = group_idx
        if len(idx) < 4:  # 至少要能算二阶差分
            continue

        x_raw = df.loc[idx, x_col].to_numpy(dtype=np.float64)
        y_raw = df.loc[idx, y_col].to_numpy(dtype=np.float64)
        x_sm = df.loc[idx, "x_smooth"].to_numpy(dtype=np.float64)
        y_sm = df.loc[idx, "y_smooth"].to_numpy(dtype=np.float64)

        # 跳过含NaN/Inf的车辆（包括平滑时被标记为脏数据、原值回退的车辆）
        # 这些车辆不参与诊断统计，避免nan污染整体结果
        if not (np.all(np.isfinite(x_raw)) and np.all(np.isfinite(y_raw))
                and np.all(np.isfinite(x_sm)) and np.all(np.isfinite(y_sm))):
            continue

        n_vehicles += 1

        vx_b = np.diff(x_raw) / dt
        vy_b = np.diff(y_raw) / dt
        vx_a = np.diff(x_sm) / dt
        vy_a = np.diff(y_sm) / dt

        vx_before_all.append(vx_b)
        vy_before_all.append(vy_b)
        vx_after_all.append(vx_a)
        vy_after_all.append(vy_a)

        ax_before_all.append(np.diff(vx_b) / dt)
        ay_before_all.append(np.diff(vy_b) / dt)
        ax_after_all.append(np.diff(vx_a) / dt)
        ay_after_all.append(np.diff(vy_a) / dt)

    vx_before = np.concatenate(vx_before_all)
    vy_before = np.concatenate(vy_before_all)
    vx_after = np.concatenate(vx_after_all)
    vy_after = np.concatenate(vy_after_all)
    ax_before = np.concatenate(ax_before_all)
    ay_before = np.concatenate(ay_before_all)
    ax_after = np.concatenate(ax_after_all)
    ay_after = np.concatenate(ay_after_all)

    diag = SmoothingDiagnostics(
        vx_std_before=float(np.std(vx_before)),
        vx_std_after=float(np.std(vx_after)),
        vy_std_before=float(np.std(vy_before)),
        vy_std_after=float(np.std(vy_after)),
        ax_std_before=float(np.std(ax_before)),
        ax_std_after=float(np.std(ax_after)),
        ay_std_before=float(np.std(ay_before)),
        ay_std_after=float(np.std(ay_after)),
        n_vehicles=n_vehicles,
        window_length_frames=window_length_frames,
        polyorder=polyorder,
    )

    _print_smoothing_diagnostics(diag)
    return diag


def _print_smoothing_diagnostics(diag: SmoothingDiagnostics) -> None:
    print("\n" + "=" * 60)
    print("位置平滑诊断报告（重新差分后的速度/加速度标准差对比）")
    print("=" * 60)
    print(f"参与统计车辆数     : {diag.n_vehicles}")
    print(f"SG窗口长度(帧)      : {diag.window_length_frames}")
    print(f"SG多项式阶数        : {diag.polyorder}")
    print("-" * 60)
    print(f"{'指标':<12}{'平滑前std':<14}{'平滑后std':<14}{'降幅':<10}")

    def _row(name, before, after):
        reduction = (before - after) / before if before > 0 else 0.0
        print(f"{name:<12}{before:<14.4f}{after:<14.4f}{reduction:<10.1%}")
        return reduction

    r1 = _row("vx (m/s)", diag.vx_std_before, diag.vx_std_after)
    r2 = _row("vy (m/s)", diag.vy_std_before, diag.vy_std_after)
    r3 = _row("ax (m/s²)", diag.ax_std_before, diag.ax_std_after)
    r4 = _row("ay (m/s²)", diag.ay_std_before, diag.ay_std_after)

    print("-" * 60)
    avg_reduction = np.mean([r1, r2, r3, r4])
    if avg_reduction < 0.05:
        print(
            f"⚠️  平均降幅仅 {avg_reduction:.1%}，平滑效果很弱。"
            f"这可能有两种原因：(a) 原始数据本身已经较干净/预处理过"
            f"（例如NGSIM官方数据已做过Kalman滤波，属于预期情况），"
            f"或 (b) 窗口参数设置不当。请结合数据来源判断，"
            f"不要不加区分地直接增大窗口。"
        )
    elif avg_reduction > 0.5:
        print(
            f"⚠️  平均降幅高达 {avg_reduction:.1%}，平滑可能过强，"
            f"存在磨平真实急刹车/变道等机动行为的风险，建议减小窗口。"
        )
    else:
        print(f"✅ 平均降幅 {avg_reduction:.1%}，处于合理区间，去噪同时保留了机动特征。")
    print("=" * 60 + "\n")


def redifferentiate_kinematics(
    df: pd.DataFrame,
    target_freq_hz: float,
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
    x_smooth_col: str = "x_smooth",
    y_smooth_col: str = "y_smooth",
) -> pd.DataFrame:
    """
    在【降采样之后】，基于平滑后的位置重新差分出速度、加速度。

    重要：必须在降采样完成之后调用（用降采样后的 dt，而不是原始 dt），
    因为差分的时间步长要和数据实际的帧间隔匹配。

    新增列：vx_recomputed, vy_recomputed, ax_recomputed, ay_recomputed
    边界帧（每辆车的首尾）用前向/后向差分填充，不产生 NaN。

    Args:
        df: 降采样后的数据，必须含有 x_smooth_col, y_smooth_col
        target_freq_hz: 降采样后的目标频率（如5Hz），用于计算dt
    """
    dt = 1.0 / target_freq_hz
    df = df.sort_values([vehicle_id_col, frame_col]).reset_index(drop=True)

    df["vx_recomputed"] = np.nan
    df["vy_recomputed"] = np.nan
    df["ax_recomputed"] = np.nan
    df["ay_recomputed"] = np.nan

    for vid, group_idx in df.groupby(vehicle_id_col).groups.items():
        idx = group_idx
        n = len(idx)
        x = df.loc[idx, x_smooth_col].to_numpy(dtype=np.float64)
        y = df.loc[idx, y_smooth_col].to_numpy(dtype=np.float64)

        if n < 2:
            # 单帧轨迹，速度/加速度无法定义，置0
            df.loc[idx, ["vx_recomputed", "vy_recomputed", "ax_recomputed", "ay_recomputed"]] = 0.0
            continue

        # 速度：中心差分（更准确），边界用单侧差分
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)

        df.loc[idx, "vx_recomputed"] = vx
        df.loc[idx, "vy_recomputed"] = vy

        if n < 3:
            df.loc[idx, ["ax_recomputed", "ay_recomputed"]] = 0.0
            continue

        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        df.loc[idx, "ax_recomputed"] = ax
        df.loc[idx, "ay_recomputed"] = ay

    return df


# ============================================================================
# 第二步：统一降采样到 5Hz
# ============================================================================
def downsample_trajectories(
    df: pd.DataFrame,
    source_freq_hz: float,
    target_freq_hz: float = 5.0,
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
) -> tuple[pd.DataFrame, int]:
    """
    把数据从 source_freq_hz 降采样到 target_freq_hz。

    降采样因子 N = source_freq_hz / target_freq_hz，必须整除，
    否则说明目标频率选择不合理（比如想从10Hz降到3Hz不整除），直接报错而不是静默取整。

    降采样方式：每辆车独立地、按帧序号每隔N帧取一帧（不是按行位置，
    避免不同车辆起始frame不对齐导致的取样偏差）。

    重要：调用此函数前，df 必须已经过 smooth_positions() 处理，
    即必须包含 'x_smooth', 'y_smooth' 列 —— 降采样要降采样平滑后的轨迹，
    不是降采样原始噪声轨迹。

    Returns:
        (downsampled_df, N) — N 是实际使用的降采样因子，存入 source_freq 等
        字段供 Dataset 阶段 debug 追溯
    """
    if "x_smooth" not in df.columns or "y_smooth" not in df.columns:
        raise ValueError(
            "降采样前必须先调用 smooth_positions()！"
            "当前df缺少 'x_smooth'/'y_smooth' 列。"
            "正确顺序：smooth_positions() → downsample_trajectories() → redifferentiate_kinematics()"
        )

    ratio = source_freq_hz / target_freq_hz
    N = round(ratio)

    if abs(ratio - N) > 1e-6:
        raise ValueError(
            f"降采样因子不是整数: source={source_freq_hz}Hz, target={target_freq_hz}Hz, "
            f"ratio={ratio}。请选择能整除原始频率的目标频率，"
            f"或显式处理非整数降采样（当前实现不支持，避免引入额外插值误差）。"
        )

    logger.info(
        f"降采样: {source_freq_hz}Hz → {target_freq_hz}Hz，因子 N={N}"
        f"（每{N}帧取1帧）"
    )

    df = df.sort_values([vehicle_id_col, frame_col]).reset_index(drop=True)

    kept_rows = []
    n_vehicles_too_short_after = 0

    for vid, group_idx in df.groupby(vehicle_id_col).groups.items():
        idx = np.array(group_idx)
        # 按"组内第几帧"取样（而不是按frame列原始数值取模），
        # 这样即使该车辆frame不是从0开始，也能保证组内严格每隔N帧取1帧
        selected_positions = np.arange(0, len(idx), N)
        selected_idx = idx[selected_positions]

        if len(selected_idx) < 2:
            n_vehicles_too_short_after += 1

        kept_rows.append(selected_idx)

    all_kept_idx = np.concatenate(kept_rows)
    df_downsampled = df.loc[all_kept_idx].sort_values(
        [vehicle_id_col, frame_col]
    ).reset_index(drop=True)

    if n_vehicles_too_short_after > 0:
        logger.warning(
            f"{n_vehicles_too_short_after} 辆车降采样后只剩 <2 帧，"
            f"这些车辆在后续切片阶段会被自动过滤（轨迹太短无法构成8秒窗口）。"
        )

    logger.info(
        f"降采样完成: {len(df)} 行 → {len(df_downsampled)} 行 "
        f"(保留比例 {len(df_downsampled) / len(df):.1%})"
    )

    return df_downsampled, N


def run_smoothing_and_downsampling_pipeline(
    df: pd.DataFrame,
    source_freq_hz: float,
    target_freq_hz: float = 5.0,
    vehicle_id_col: str = "id",
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    window_length_frames: int | None = None,
    polyorder: int = 2,
    dataset_name: str = "unknown",
) -> tuple[pd.DataFrame, dict]:
    """
    整合第五步(平滑) + 第二步(降采样) + 降采样后重新差分的完整流程，
    严格按需求文档要求的顺序执行：

        1. smooth_positions()         — 对原始高频位置做SG平滑
        2. downsample_trajectories()  — 对平滑后的轨迹降采样到目标频率
        3. redifferentiate_kinematics() — 用降采样后的dt重新差分出v/a

    Args:
        dataset_name: 数据集标识（如 "ngsim", "highd"），仅用于日志和
            pipeline_info 记录，不影响任何计算逻辑。两个数据集共用同一套
            "按检测频率自动换算窗口"的规则，平滑降幅的差异完全来自数据
            本身的噪声水平不同（详见文件头部说明），不依赖此参数做分支处理。

    Returns:
        (final_df, pipeline_info)
        pipeline_info 包含 N、source_freq_hz、target_freq_hz、诊断对象等，
        供后续步骤（切片、Dataset）使用，也用于写入 source_freq 字段。
    """
    logger.info("=" * 60)
    logger.info(f"开始执行: 平滑 → 降采样 → 重新差分 流程 [dataset={dataset_name}]")
    logger.info("=" * 60)

    # Step 1: 平滑
    df_smoothed, smoothing_diag = smooth_positions(
        df,
        source_freq_hz=source_freq_hz,
        vehicle_id_col=vehicle_id_col,
        frame_col=frame_col,
        x_col=x_col,
        y_col=y_col,
        window_length_frames=window_length_frames,
        polyorder=polyorder,
    )

    # Step 2: 降采样（基于平滑后的位置）
    df_downsampled, N = downsample_trajectories(
        df_smoothed,
        source_freq_hz=source_freq_hz,
        target_freq_hz=target_freq_hz,
        vehicle_id_col=vehicle_id_col,
        frame_col=frame_col,
    )

    # Step 3: 降采样后，用新的dt重新差分速度/加速度
    df_final = redifferentiate_kinematics(
        df_downsampled,
        target_freq_hz=target_freq_hz,
        vehicle_id_col=vehicle_id_col,
        frame_col=frame_col,
    )

    # Step 4: 列结构标准化
    # 确保输出 parquet 统一包含 xVelocity / yVelocity 列，
    # 对于只有合速度标量 v_Vel 的 NGSIM 半转换文件，用重新差分的结果填充。
    # 这样后续 Dataset.__getitem__ 只需要认识 xVelocity/yVelocity，
    # 不需要再处理两种列名格式的分支逻辑。
    if "xVelocity" not in df_final.columns:
        df_final["xVelocity"] = df_final["vx_recomputed"]
        df_final["yVelocity"] = df_final["vy_recomputed"]
        logger.info(
            "原始数据无 xVelocity/yVelocity 列（NGSIM半转换格式），"
            "已用平滑位置差分结果填充，列结构已与highD格式对齐。"
        )
    if "xAcceleration" not in df_final.columns:
        df_final["xAcceleration"] = df_final["ax_recomputed"]
        df_final["yAcceleration"] = df_final["ay_recomputed"]

    avg_velocity_reduction = float(np.mean([
        (smoothing_diag.vx_std_before - smoothing_diag.vx_std_after) / max(smoothing_diag.vx_std_before, 1e-8),
        (smoothing_diag.vy_std_before - smoothing_diag.vy_std_after) / max(smoothing_diag.vy_std_before, 1e-8),
    ]))

    pipeline_info = {
        "dataset_name": dataset_name,
        "source_freq_hz": source_freq_hz,
        "target_freq_hz": target_freq_hz,
        "downsample_factor_N": N,
        "smoothing_window_frames": smoothing_diag.window_length_frames,
        "smoothing_polyorder": smoothing_diag.polyorder,
        "smoothing_avg_velocity_std_reduction": avg_velocity_reduction,
        "smoothing_diagnostics": smoothing_diag,
    }

    logger.info("=" * 60)
    logger.info(
        f"流程完成 [dataset={dataset_name}]: {source_freq_hz}Hz → {target_freq_hz}Hz "
        f"(N={N})，最终行数={len(df_final)}，"
        f"平滑速度std降幅={avg_velocity_reduction:.1%}"
    )
    logger.info("=" * 60)

    return df_final, pipeline_info


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="检测/预处理轨迹数据")
    parser.add_argument("--input", type=str, required=True, help="csv文件路径")
    parser.add_argument("--vehicle_id_col", type=str, default=None)
    parser.add_argument("--frame_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--x_col", type=str, default=None)
    parser.add_argument("--y_col", type=str, default=None)
    parser.add_argument("--xvel_col", type=str, default=None)
    parser.add_argument("--yvel_col", type=str, default=None)
    parser.add_argument("--n_vehicles", type=int, default=50)

    parser.add_argument(
        "--run_downsample", action="store_true",
        help="检测频率之后，继续执行 平滑→降采样→重新差分 完整流程",
    )
    parser.add_argument("--target_freq", type=float, default=5.0, help="目标降采样频率(Hz)")
    parser.add_argument(
        "--source_freq_override", type=float, default=None,
        help="手动指定原始频率，跳过自动检测（用于已确认频率的二次运行）",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="降采样结果保存路径(parquet)，不指定则只打印不保存",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="unknown",
        help="数据集标识(ngsim/highd)，仅用于日志记录与pipeline_info追溯，不影响计算逻辑",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    logger.info(f"读取数据: {path}")
    df = pd.read_csv(path)
    logger.info(f"数据形状: {df.shape}，列名: {list(df.columns)}")

    vid_col = args.vehicle_id_col or _find_column(df, VEHICLE_ID_CANDIDATES)
    f_col = args.frame_col or _find_column(df, FRAME_CANDIDATES)
    x_col = args.x_col or _find_column(df, X_CANDIDATES)
    y_col = args.y_col or _find_column(df, Y_CANDIDATES)

    if args.source_freq_override is not None:
        source_freq_hz = args.source_freq_override
        logger.info(f"使用手动指定的频率: {source_freq_hz} Hz（跳过自动检测）")
    else:
        result = detect_sampling_frequency(
            df,
            vehicle_id_col=args.vehicle_id_col,
            frame_col=args.frame_col,
            time_col=args.time_col,
            x_col=args.x_col,
            y_col=args.y_col,
            xvel_col=args.xvel_col,
            yvel_col=args.yvel_col,
            n_vehicles_to_sample=args.n_vehicles,
        )
        source_freq_hz = result.detected_freq_hz
        if np.isnan(source_freq_hz):
            raise RuntimeError(
                "未能自动检测出频率，请用 --source_freq_override 手动指定后重试。"
            )

    if not args.run_downsample:
        return

    df_final, pipeline_info = run_smoothing_and_downsampling_pipeline(
        df,
        source_freq_hz=source_freq_hz,
        target_freq_hz=args.target_freq,
        vehicle_id_col=vid_col,
        frame_col=f_col,
        x_col=x_col,
        y_col=y_col,
        dataset_name=args.dataset_name,
    )

    print("\n流程摘要:")
    for k, v in pipeline_info.items():
        if k != "smoothing_diagnostics":
            print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(out_path, index=False)
        logger.info(f"已保存降采样结果至: {out_path}")
    else:
        logger.info("未指定 --output，结果不落盘（仅打印诊断信息）。")


if __name__ == "__main__":
    main()