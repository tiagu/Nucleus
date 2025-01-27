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

pip install uv

uv venv /nemo/lab/briscoej/home/users/ritot/nucleus_t18/nucleus-2 --seed --python python3.9

source nucleus-2/bin/activate

uv pip install torch==1.10 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

uv pip install jupyterlab

uv pip install --upgrade pip

# deal with new incompatible versions
uv pip install Pillow==9.5
uv pip install numpy==1.21.6 contourpy matplotlib scikit-image scipy pandas opencv-python tqdm

python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

unset XDG_RUNTIME_DIR; python -m jupyterlab --ip=$(hostname -I | awk '{print $1}') --port=8000 /camp/home/ritot/home/users/ritot/  | tee jupyter.$(hostname -I | awk '{print $1}').output


