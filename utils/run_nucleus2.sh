#!/bin/bash
#SBATCH --time=2-23:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --partition=ga100
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-nucleus2-%j.out

# submit as sbatch run_nucleus.sh 
ml foss
ml Anaconda3

source nucleus-2/bin/activate

unset XDG_RUNTIME_DIR; python -m jupyterlab --ip=$(hostname -I | awk '{print $1}') --port=8000 /camp/home/  | tee jupyter.$(hostname -I | awk '{print $1}').output


