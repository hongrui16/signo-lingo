#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=python-gpu
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/logs/comp/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/logs/comp/%u/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8                # number of cores needed
#SBATCH --gres=gpu:3g.40gb:1                # up to 8; only request what you need
#SBATCH --mem=40gb                # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-24:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10                           
module load python

source /home/rhong5/py39torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src

# python compute_3D_kpts.py --dataset_name AUTSL
python compute_3D_kpts.py --dataset_name WLASL


# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00