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

echo "Start: $(date)"

#python scripts/batch_preprocess.py \
#  --ngsim_dir  data/raw/ngsim \
#  --highd_dir  data/raw/highD \
#  --output_dir data/processed \
#  --dataset    highd \
#  --target_freq 5.0 \
#  --resume
#
#python scripts/batch_preprocess.py \
#  --ngsim_dir  data/raw/ngsim \
#  --highd_dir  data/raw/highD \
#  --output_dir data/processed \
#  --dataset    highd \
#  --target_freq 5.0 \
#  --resume


python -m src.data.feature_precompute \
  --index    data/processed/global_window_index.parquet \
  --proc_dir data/processed \
  --output   data/processed/features_cache.parquet \
  --dataset  highd \
  --batch    10000

echo "Done: $(date)"