#!/bin/sh
#SBATCH --job-name=cs265
#SBATCH --partition gpu_requeue
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --mem=32G
#SBATCH -t 0-6:00 
#SBATCH -c 4
#SBATCH --output=/n/home04/keremdayi/logs/cs265_%A__%a.out
#SBATCH --mail-user=keremdayi@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load intel-mkl/24.2.1-fasrc01

mamba activate cs265

export GLOO_SOCKET_IFNAME=eth0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

cd /n/home04/keremdayi/CS265-mlsys-project
python starter_code.py