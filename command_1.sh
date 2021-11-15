#!/bin/sh

#SBATCH --job-name=parser_gpu16
#SBATCH --qos=csqos
#SBATCH --output=/scratch/%u/out_files/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/scratch_files/%x-%N-%j.err   # Error file
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=trahman2@gmu.edu     # Put your GMU email address here
#SBATCH --mem=200G    # Total memory needed per task (units: K,M,G,T)
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2    # Number of GPUs needed
##SBATCH --nodelist=NODE078  # If you want to run on a specific node
#SBATCH --nodes=1
#SBATCH --tasks=1

## Run your program or script
module purge
module load cuda/10.2
module load gcc/8.4.0
##source ~/torch-with-cuda/bin/activate

python test.py

##python vanilla_6_ttur.py
