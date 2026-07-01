#!/bin/bash
#SBATCH --job-name=dg_preprocess
#SBATCH --partition=multicore
#SBATCH --ntasks=8
#SBATCH --mem=64G               # 66个文件批量读取，预留充足内存
#SBATCH --time=4-0                  # 最大4天（实际应该2-4小时能跑完）
#SBATCH --output=slurm/logs/%j_preprocess.out
#SBATCH --error=slurm/logs/%j_preprocess.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xinyue.liu-9@postgrad.manchester.ac.uk

# -----------------------------------------------------------------------
# 环境初始化
# -----------------------------------------------------------------------
module load apps/binapps/anaconda3/2023.09
#source activate TAV

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Work dir   : $(pwd)"
echo "=========================================="

# -----------------------------------------------------------------------
# 第一步：只处理NGSIM（6个文件，先跑通，快速验证）
# -----------------------------------------------------------------------
echo ""
echo "[STEP 1] 处理NGSIM文件..."
python scripts/batch_preprocess.py \
    --ngsim_dir  data/raw/ngsim \
    --highd_dir  data/raw/highD \
    --output_dir data/processed \
    --dataset    ngsim \
    --target_freq 5.0 \
    --resume

echo ""
echo "[STEP 1 DONE] NGSIM处理完成: $(date)"

# -----------------------------------------------------------------------
# 第二步：处理highD（60个文件，时间较长）
# -----------------------------------------------------------------------
echo ""
echo "[STEP 2] 处理highD文件..."
python scripts/batch_preprocess.py \
    --ngsim_dir  data/raw/ngsim \
    --highd_dir  data/raw/highD \
    --output_dir data/processed \
    --dataset    highd \
    --target_freq 5.0 \
    --resume

echo ""
echo "[STEP 2 DONE] highD处理完成: $(date)"

# -----------------------------------------------------------------------
# 第三步：合并全局索引（ALL模式，会读已缓存的结果，基本只做合并）
# -----------------------------------------------------------------------
echo ""
echo "[STEP 3] 合并全局窗口索引..."
python scripts/batch_preprocess.py \
    --ngsim_dir  data/raw/ngsim \
    --highd_dir  data/raw/highD \
    --output_dir data/processed \
    --dataset    all \
    --target_freq 5.0 \
    --resume

echo ""
echo "=========================================="
echo "全部完成: $(date)"
echo "全局窗口索引: data/processed/global_window_index.parquet"
echo "=========================================="