#!/bin/bash
#SBATCH -p gpu_titanrtx
#SBATCH -t 06:00:00

# Run me with (update name!)   sbatch --gpus-per-node=4 vilBERT_imdb_less_self_super.sh

# Set environment variables
echo "Setting environment variables..."
export RUN_NAME="vilBERT_imdb_less_self_super_20k"
export _ROOT_=$HOME"/new_root"

# Environment variables for MMF
export MMF_CACHE_DIR="$TMPDIR"/thomass_scratch/home/.cache/torch
export MMF_DATA_DIR="$TMPDIR"/thomass_scratch/home/.cache/torch/mmf/data
export MMF_SAVE_DIR="$TMPDIR"/thomass_scratch/save

echo $RUN_NAME
echo "Checks:"
echo "TMPDIR: $TMPDIR"
echo "HOME: $HOME"

# Load modules
echo "Loading modules..."
module load 2020
module load Anaconda3

# Check Nvidia on compute node
echo "Checking CUDA..."
nvcc -V
nvidia-smi

# Move data to scratch
echo "Moving data to scratch..."
mkdir -p "$TMPDIR"/thomass_scratch/save
# use -rn if only new files should be added
cp -r $_ROOT_/home "$TMPDIR"/thomass_scratch/
chmod -R a+rwX "$TMPDIR"/thomass_scratch
cd "$TMPDIR"/thomass_scratch/home/.cache/torch/mmf/data/datasets

echo "unzipping without overwriting..."
unzip -qn mmimdb.zip && unzip -q mmimdb_self_super.zip

# !!! CHOSE ONE !!!
# option 1: one of the label splits labels.jsonl or no_labels.jsonl
mv mmimdb_self_super/no_labels.jsonl mmimdb/defaults/annotations/train.jsonl
# option 2: original full train.jsonl
#mv mmimdb_full_train.jsonl mmimdb/defaults/annotations/train.jsonl

cd $HOME

# Prepare output folder
echo "Preparing output folder..."
mkdir $HOME/job_saves/$RUN_NAME
chmod -R a+rwX $HOME/job_saves/$RUN_NAME

# Setup conda environment
echo "Setting up conda environment..."
cd $_ROOT_/opt/conda/bin && source activate
conda activate $_ROOT_/opt/conda/envs/mmf

# No matter what happens, we copy the save folder to our login node
trap 'cp -r $TMPDIR/thomass_scratch/save $HOME/job_saves/$RUN_NAME' EXIT

echo "Moving checkpoint to scratch..."
# Move checkpoint to scratch
cp -r $HOME/job_saves/$RUN_NAME/save "$TMPDIR"/thomass_scratch/

# Making sure all gpus are used
unset CUDA_VISIBLE_DEVICES

# Run
## Train without val in pretraining? Disabled eval...
mmf_run config=projects/vilbert/configs/mmimdb/pretrain.yaml \
    run_type=train \
    dataset=masked_mmimdb \
    model=vilbert \
    training.fp16=True \
    training.tensorboard=True \
    training.batch_size=80 \
    training.update_frequency=1 \
    training.checkpoint_interval=500 \
    training.max_updates=20000 \
    scheduler.params.num_training_steps=20000 \
    training.clip_gradients=True \
    training.max_grad_l2_norm=0.25 \
    training.seed=45003754 \
    training.find_unused_parameters=True \
    training.evaluation_interval=100_000 \
    checkpoint.resume=True \
    checkpoint.max_to_keep=2
