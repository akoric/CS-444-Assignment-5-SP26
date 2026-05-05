#!/bin/bash
#SBATCH --job-name=latent-diffusion-vae
#SBATCH --account=bbkg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=/u/akoric/myproject/assignment_5_starter_code/batch_outputs/latent-diffusion-%j.out
#SBATCH --error=/u/akoric/myproject/assignment_5_starter_code/batch_outputs/latent-diffusion-%j.err

set -euo pipefail

OUTPUT_DIR=/u/akoric/myproject/assignment_5_starter_code/batch_outputs
mkdir -p "$OUTPUT_DIR"

module load python/anaconda3/2.10.0
export CONDA_PKGS_DIRS=/u/$USER/.conda/pkgs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /u/akoric/envs/cs444-mp4

cd /u/akoric/myproject/assignment_5_starter_code

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "GPU:"
nvidia-smi -L

python train_latent_diffusion.py

echo "Finished:  $(date)"
