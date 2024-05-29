#!/bin/bash
#SBATCH --time=2-23:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-nucleus-%j.out


# submit as sbatch run_nucleus.sh 


ml Anaconda3/2019.07

source activate nucleus

python -m ipykernel install --user --name=nucleus_t


config_dir=$(jupyter --config-dir)
echo "Jupyter configuration directory: $config_dir"


unset XDG_RUNTIME_DIR; python -m notebook --ip=$(hostname -I | awk '{print $1}') --port=8000 /nemo/lab/path/to/folder/nucleus | tee jupyter.$(hostname -I | awk '{print $1}').output
