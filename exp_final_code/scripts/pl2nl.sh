#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH --job-name="CodeCG"
#SBATCH -o /scratch/bhanu/CodeCG-exp/outs/pl2nl-%j.out
#SBATCH -e /scratch/bhanu/CodeCG-exp/outs/pl2nl-%j.err
#SBATCH --mail-user=f20190083@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=NONE

# Load modules
spack unload 
spack load gcc@11.2.0
spack load cuda@11.7.1%gcc@11.2.0 arch=linux-rocky8-zen2
spack load python@3.9.13%gcc@11.2.0 arch=linux-rocky8-zen2


# Activate Environment
source ~/CodeCG/codecg/bin/activate

# Run 
cd ~/CodeCG/exp-final-code

srun ~/CodeCG/codecg/bin/python main.py \
--run_name pl2nl-$SLURM_JOB_ID \
--jobid $SLURM_JOB_ID \
--logger wandb \
--path_logs "/scratch/bhanu/CodeCG-exp/logs" \
--path_base_models "/scratch/bhanu/CodeCG-exp/base_models" \
--path_cache_datasets "/scratch/bhanu/CodeCG-exp/dataset" \
--nl_de_model "/scratch/bhanu/CodeCG-exp/saved_models/nl-decoder--JOB_ID" \
--pl_en_model "/scratch/bhanu/CodeCG-exp/saved_models/pl-encoder-26128" \
--workers 4 \
--epochs 3 \
--batch_size 128 \
