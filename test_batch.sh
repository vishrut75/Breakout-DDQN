#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem 6G
#SBATCH --gres=gpu:1
#SBATCH -o test_v_1.txt
#SBATCH -t 20:00:00

echo "starting python"
python3 main.py --test_dqn &
wait

