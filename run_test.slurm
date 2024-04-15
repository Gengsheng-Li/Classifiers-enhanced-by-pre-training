#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --nodelist gpu01
#SBATCH -o ./slurm_log/slurm%A.log
#SBATCH -e ./slurm_log/slurm%A.log

python test.py --model_pth 'results/fine-tune-best.pth' \