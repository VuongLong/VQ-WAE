#!/bin/bash

#SBATCH --job-name=swagsam.py

#SBATCH --output=/lustre/scratch/client/vinai/users/longvt8/log_LASSO/pa%A.out

#SBATCH --error=/lustre/scratch/client/vinai/users/longvt8/log_LASSO/pa%A.err

#SBATCH --gpus=1

#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G

#SBATCH --cpus-per-gpu=32

#SBATCH --partition=research
#SBATCH --mail-type=all #SBATCH --mail-user=v.longvt8@vinai.io

module purge

eval "$(conda shell.bash hook)"

source activate /lustre/scratch/client/vinai/users/longvt8/miniconda3/
export PYTHONPATH=$PWD

python main.py -c "celeba_fast_vqwae_C512.yaml" --save --gpu 0 --dbg