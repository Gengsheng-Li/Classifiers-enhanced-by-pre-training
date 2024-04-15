#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --nodelist gpu01
#SBATCH -o ./slurm_log/slurm%A.log
#SBATCH -e ./slurm_log/slurm%A.log

python train_from_pretrain.py --train_mode 'fine-tune' \
                            --lr 1e-6 \
                            --epochs 100 \
                            --batch_size 512 \
                            --validation_split 0.2 \
                            --save_path "results" \
                            --exp_name "train_from_pretrain" \