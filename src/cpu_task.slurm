#!/bin/bash
#SBATCH --partition=normal                           
#SBATCH --job-name=download_data
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/logs/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src/logs/%u/%x-%N-%j.err    # Error file
#SBATCH --time=1-23:00:00                   # set to 2hr; please choose carefully
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2                # number of cores needed
#SBATCH --mem-per-cpu=2GB             # Request nGB RAM per core
#SBATCH --export=ALL 

set echo
umask 0027


module load gnu10                           
module load python

source /home/rhong5/py39torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/signo-lingo/src



python preprocess.py
