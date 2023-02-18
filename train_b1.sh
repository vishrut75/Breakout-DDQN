#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH --gres=gpu:1
#SBATCH -o train_v_2.txt
#SBATCH -t 20:00:00

echo "starting python"
python3 main.py --train_dqn &
wait

