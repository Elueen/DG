"""
src/data/dataset.py

PyTorch Dataset — 数据侧最终出口。

__getitem__ 接收一个窗口索引，从对应的 *_5hz.parquet 切出数值，
再从 features_cache.parquet 读取预计算的邻车/TTC/变道特征，
打包成模型所需的张量字典返回。

返回字典结构（对应需求文档）：
{
  'hist':              [N_agents=7, T_hist=15, 4],   # ego + 6邻车，(x,y,vx,vy)
  'fut':               [T_fut=25, 2],                # ego未来轨迹 (x,y)
  'mask_fixed6':       [6],                           # 邻车有效性mask
  'interaction_feats': [6, T_hist=15, K=6],          # TTC/THW/rel_vx/rel_vy等
  'valid_ttc_mask':    [6],                           # TTC是否有效
  'is_lane_change':    scalar bool tensor
  'scene_density':     scalar int tensor
  'source_freq':       scalar float tensor            # 原始采样频率，debug用
  'window_id':         str                            # 唯一标识，非张量
}

N_agents=7: ego(1) + 最多6辆邻车，ego 始终放在索引0。
hist 中无效邻车槽位的值全为0，配合 mask_fixed6 使用。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 社交网格槽位顺序（与 feature_precompute.py 保持一致）
SLOT_NAMES = [
    "left_preceding", "left_alongside", "left_following",
    "right_preceding", "right_alongside", "right_following",
]
N_SLOTS   = 6
T_HIST    = 15   # 3秒 × 5Hz
T_FUT     = 25   # 5秒 × 5Hz
N_AGENTS  = 7    # ego + 6邻车

# interaction_feats 的 K 个维度（顺序固定，方便模型侧解包）
# [ttc, thw, rel_vx, rel_vy, valid_ttc, mask]
INTERACTION_K = 6


class TrajectoryDataset(Dataset):
    """
    多智能体轨迹预测 Dataset。

    Args:
        split_index_path: train.parquet / val.parquet / test.parquet 路径
            （由 split.py 产出，是 global_window_index 的子集）
        processed_dir: 降采样后的 *_5hz.parquet 所在根目录
        features_cache_path: feature_precompute.py 产出的特征缓存 parquet
        normalize: 是否对轨迹坐标做局部归一化
            （以历史段最后一帧 ego 位置为原点，以 ego 速度方向为 x 轴）
        cache_in_memory: 是否把所有 *_5hz.parquet 预加载到内存
            （小数据集推荐True，大数据集按需读取）
    """

    def __init__(
        self,
        split_index_path: str | Path,
        processed_dir: str | Path,
        features_cache_path: str | Path,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        self.processed_dir      = Path(processed_dir)
        self.normalize          = normalize
        self.cache_in_memory    = cache_in_memory

        # 读取窗口索引（该 split 的所有窗口元数据）
        logger.info(f"读取 split 索引: {split_index_path}")
        self.index_df = pd.read_parquet(split_index_path).reset_index(drop=True)
        logger.info(f"  共 {len(self.index_df)} 个窗口")

        # 读取特征缓存，以 window_id 为主键建立快速查找
        logger.info(f"读取特征缓存: {features_cache_path}")
        feat_df = pd.read_parquet(features_cache_path)
        self.feat_lookup: dict[str, dict] = {
            row["window_id"]: row.to_dict()
            for _, row in feat_df.iterrows()
        }
        logger.info(f"  特征缓存覆盖 {len(self.feat_lookup)} 个窗口")

        # 可选：预加载所有 parquet 到内存
        self._file_cache: dict[str, pd.DataFrame] = {}
        if cache_in_memory:
            self._preload_all_files()

    def _preload_all_files(self) -> None:
        """把所有涉及的 *_5hz.parquet 预加载到内存字典。"""
        file_tags = self.index_df["source_file"].unique()
        logger.info(f"预加载 {len(file_tags)} 个 parquet 文件到内存...")
        for tag in file_tags:
            dataset_name = self.index_df[
                self.index_df["source_file"] == tag
            ]["dataset_name"].iloc[0]
            path = self.processed_dir / dataset_name / f"{tag}_5hz.parquet"
            if path.exists():
                self._file_cache[tag] = pd.read_parquet(path)
        logger.info("预加载完成")

    def _get_file_df(self, source_file: str, dataset_name: str) -> pd.DataFrame:
        """按需读取 parquet（有缓存就用缓存）。"""
        if source_file not in self._file_cache:
            path = self.processed_dir / dataset_name / f"{source_file}_5hz.parquet"
            self._file_cache[source_file] = pd.read_parquet(path)
        return self._file_cache[source_file]

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> dict:
        meta  = self.index_df.iloc[idx]
        wid   = meta["window_id"]
        src   = meta["source_file"]
        dname = meta["dataset_name"]
        ego_id= meta["vehicle_id"]
        s_fr  = int(meta["start_frame"])
        e_fr  = int(meta["end_frame"])

        df = self._get_file_df(src, dname)

        # ── ego 轨迹切片 ──────────────────────────────────────────────
        ego_sorted = df[df["id"] == ego_id].sort_values("frame").reset_index(drop=True)
        ego_window = ego_sorted.iloc[s_fr:e_fr]

        if len(ego_window) != (e_fr - s_fr):
            # 数据不完整，返回全零样本（训练时 DataLoader 会跳过或被 collate 处理）
            return self._zero_sample(wid)

        ego_hist = ego_window.iloc[:T_HIST]   # 前15帧
        ego_fut  = ego_window.iloc[T_HIST:]   # 后25帧

        # ego 历史：(x, y, vx, vy) × 15帧
        ego_hist_arr = np.stack([
            ego_hist["x_smooth"].to_numpy(dtype=np.float32),
            ego_hist["y_smooth"].to_numpy(dtype=np.float32),
            ego_hist["xVelocity"].to_numpy(dtype=np.float32),
            ego_hist["yVelocity"].to_numpy(dtype=np.float32),
        ], axis=-1)  # [T_hist, 4]

        # ego 未来：(x, y) × 25帧
        ego_fut_arr = np.stack([
            ego_fut["x_smooth"].to_numpy(dtype=np.float32),
            ego_fut["y_smooth"].to_numpy(dtype=np.float32),
        ], axis=-1)  # [T_fut, 2]

        # ── 邻车轨迹（从特征缓存取 ID，再切轨迹）────────────────────
        feat = self.feat_lookup.get(wid, {})
        hist_frame_val = ego_hist.iloc[-1]["frame"]  # 历史末帧的原始 frame 值
        same_frame_df  = df[df["frame"] == hist_frame_val]

        neigh_hist_arr  = np.zeros((N_SLOTS, T_HIST, 4), dtype=np.float32)
        mask_fixed6     = np.zeros(N_SLOTS, dtype=np.float32)
        interact_feats  = np.zeros((N_SLOTS, T_HIST, INTERACTION_K), dtype=np.float32)
        valid_ttc_mask  = np.zeros(N_SLOTS, dtype=np.float32)

        for si, slot in enumerate(SLOT_NAMES):
            if feat.get(f"{slot}_mask", 0) != 1:
                continue  # 该槽位无车

            other_id = feat.get(f"{slot}_id")
            if other_id is None or np.isnan(other_id):
                continue

            # 取邻车历史轨迹（同样按组内位置切片）
            other_sorted = df[df["id"] == other_id].sort_values("frame").reset_index(drop=True)
            if len(other_sorted) < e_fr:
                # 邻车在该窗口内没有足够帧数，用最近帧广播填充（保守处理）
                other_hist_frames = other_sorted.iloc[max(0, len(other_sorted)-T_HIST):]
            else:
                other_hist_frames = other_sorted.iloc[s_fr:s_fr+T_HIST]

            if len(other_hist_frames) == 0:
                continue

            # 填充到固定长度 T_HIST（不足时前向广播）
            n_avail = len(other_hist_frames)
            oh = np.zeros((T_HIST, 4), dtype=np.float32)
            arr = np.stack([
                other_hist_frames["x_smooth"].to_numpy(dtype=np.float32),
                other_hist_frames["y_smooth"].to_numpy(dtype=np.float32),
                other_hist_frames["xVelocity"].to_numpy(dtype=np.float32),
                other_hist_frames["yVelocity"].to_numpy(dtype=np.float32),
            ], axis=-1)
            oh[T_HIST - n_avail:] = arr[:T_HIST]
            if n_avail < T_HIST:
                oh[:T_HIST - n_avail] = arr[0]  # 前向复制最早帧

            neigh_hist_arr[si] = oh
            mask_fixed6[si] = 1.0

            # interaction_feats: [ttc, thw, rel_vx, rel_vy, valid_ttc, mask]
            ttc      = feat.get(f"{slot}_ttc", np.nan)
            thw      = feat.get(f"{slot}_thw", np.nan)
            rel_vx   = feat.get(f"{slot}_rel_vx", np.nan)
            rel_vy   = feat.get(f"{slot}_rel_vy", np.nan)
            ttc_ok   = float(feat.get(f"{slot}_ttc_valid", 0))

            # 把标量特征广播到整段历史（简化：用历史末帧的快照值填满T_HIST）
            # 更精确的做法是逐帧计算，留作 v2 优化
            interact_feats[si, :, 0] = 0.0 if np.isnan(ttc)    else ttc
            interact_feats[si, :, 1] = 0.0 if np.isnan(thw)    else thw
            interact_feats[si, :, 2] = 0.0 if np.isnan(rel_vx) else rel_vx
            interact_feats[si, :, 3] = 0.0 if np.isnan(rel_vy) else rel_vy
            interact_feats[si, :, 4] = ttc_ok
            interact_feats[si, :, 5] = 1.0  # mask（有效邻车）
            valid_ttc_mask[si] = ttc_ok

        # ── 拼合 hist：[N_agents=7, T_hist, 4]，ego 在索引0 ──────────
        # neigh_hist_arr: [6, T_hist, 4]
        hist_all = np.concatenate(
            [ego_hist_arr[np.newaxis], neigh_hist_arr], axis=0
        )  # [7, 15, 4]

        # ── 局部坐标归一化 ────────────────────────────────────────────
        if self.normalize:
            hist_all, ego_fut_arr = self._normalize(hist_all, ego_fut_arr)

        # ── 打包张量字典 ──────────────────────────────────────────────
        return {
            "hist":              torch.from_numpy(hist_all),
            "fut":               torch.from_numpy(ego_fut_arr),
            "mask_fixed6":       torch.from_numpy(mask_fixed6),
            "interaction_feats": torch.from_numpy(interact_feats),
            "valid_ttc_mask":    torch.from_numpy(valid_ttc_mask),
            "is_lane_change":    torch.tensor(feat.get("is_lane_change", 0), dtype=torch.bool),
            "scene_density":     torch.tensor(feat.get("scene_density", 0),  dtype=torch.long),
            "source_freq":       torch.tensor(float(meta.get("source_freq_hz", 10.0)), dtype=torch.float32),
            "window_id":         wid,  # 字符串，不转 tensor，collate 时单独处理
        }

    @staticmethod
    def _normalize(
        hist: np.ndarray,        # [N_agents, T_hist, 4]
        fut:  np.ndarray,        # [T_fut, 2]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        局部坐标归一化：以 ego 历史段最后一帧为原点，
        ego 行驶方向（vx,vy）为 x 轴，做平移+旋转。

        好处：让模型不需要学习绝对坐标的含义，只需要学习相对运动关系。
        """
        # ego 历史末帧的位置和速度方向
        origin_x = hist[0, -1, 0]  # ego，最后一帧，x
        origin_y = hist[0, -1, 1]  # ego，最后一帧，y
        vx       = hist[0, -1, 2]
        vy       = hist[0, -1, 3]
        speed    = np.sqrt(vx**2 + vy**2) + 1e-6

        # 旋转矩阵：把 (vx,vy) 方向旋转到 x 轴正方向
        cos_h =  vx / speed
        sin_h =  vy / speed
        R = np.array([[cos_h, sin_h],
                      [-sin_h, cos_h]], dtype=np.float32)  # [2,2]

        def _transform_pos(pos_xy: np.ndarray) -> np.ndarray:
            """平移 + 旋转 (N, 2) 或 (T, 2) 的位置数组。"""
            shifted = pos_xy - np.array([origin_x, origin_y], dtype=np.float32)
            return (R @ shifted.T).T

        def _transform_vel(vel_xy: np.ndarray) -> np.ndarray:
            """只旋转，不平移，(N, 2) 或 (T, 2) 的速度数组。"""
            return (R @ vel_xy.T).T

        # 对所有智能体的历史轨迹做变换
        hist_out = hist.copy()
        for i in range(hist_out.shape[0]):  # 遍历 N_agents
            hist_out[i, :, :2] = _transform_pos(hist_out[i, :, :2])
            hist_out[i, :, 2:] = _transform_vel(hist_out[i, :, 2:])

        # 对 ego 未来轨迹做变换（只有位置）
        fut_out = _transform_pos(fut.copy())

        return hist_out, fut_out

    def _zero_sample(self, wid: str) -> dict:
        """数据不完整时返回全零占位样本（DataLoader 可通过 collate_fn 过滤）。"""
        return {
            "hist":              torch.zeros(N_AGENTS, T_HIST, 4),
            "fut":               torch.zeros(T_FUT, 2),
            "mask_fixed6":       torch.zeros(N_SLOTS),
            "interaction_feats": torch.zeros(N_SLOTS, T_HIST, INTERACTION_K),
            "valid_ttc_mask":    torch.zeros(N_SLOTS),
            "is_lane_change":    torch.tensor(False),
            "scene_density":     torch.tensor(0, dtype=torch.long),
            "source_freq":       torch.tensor(10.0),
            "window_id":         wid,
            "_is_zero_sample":   True,  # 标记，collate_fn 可用来过滤
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    自定义 collate 函数，处理两个特殊情况：
    1. 过滤掉 _zero_sample（数据不完整的窗口）
    2. 把 window_id（字符串列表）和张量字段分开处理
    """
    # 过滤零样本
    batch = [b for b in batch if not b.get("_is_zero_sample", False)]
    if not batch:
        return {}

    window_ids = [b.pop("window_id") for b in batch]

    # 标准张量字段 stack
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    out["window_id"] = window_ids
    return out