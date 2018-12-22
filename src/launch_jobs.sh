# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset.
# 2. Trains an unconditional, conditional, or InfoGAN model on the MNIST
#    training set.
# 3. Evaluates the models and writes sample images to disk.
#
# These examples are intended to be fast. For better final results, tune
# hyperparameters or train longer.
#
# NOTE: Each training step takes about 0.5 second with a batch size of 32 on
# CPU. On GPU, it takes ~5 milliseconds.
#
# With the default batch size and number of steps, train times are:
#
#   unconditional: CPU: 800  steps, ~10 min   GPU: 800  steps, ~1 min
#   conditional:   CPU: 2000 steps, ~20 min   GPU: 2000 steps, ~2 min
#   infogan:       CPU: 3000 steps, ~20 min   GPU: 3000 steps, ~6 min
#
# Usage:
# ./launch_jobs.sh ${run_mode} ${version}
set -e

# define run mode
run_mode=$1
if ! [[ "$run_mode" =~ ^(test|training|triplet_training|visualization) ]]; then
    echo "'run_mode' must be one of: 'test', 'training', 'triplet_training', 'visualization'."
    exit
fi

# define version for the log files
version=$2
if [[ "$version" == "" ]]; then
    version="test"
fi

# define number of steps to run the experiment
NUM_EPOCHS=$3
if [[ "$NUM_EPOCHS" == "" ]]; then
    NUM_EPOCHS=1
fi

# define which GPU to run on
gpu_unit=$4
if [[ "$gpu_unit" == "" ]]; then
    echo "use default gpu (GPU0)."
    gpu_unit=0
fi

export CUDA_VISIBLE_DEVICES=$gpu_unit
PYTHON="python"

# Location of the git repository.
git_repo="../Tensorflow-models"

# Location of the src directory
src_dir="../TripletSent/src"

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=ts-models/${version}

# Base name for where the evaluation images will be saved to.
EVAL_DIR=ts-models/eval/${version}

# Where the dataset is saved to.
# DATASET_DIR=datasets/sentibank_flickr/regular_128/tfrecord
# DATASET_DIR=datasets/sentibank_flickr/regular_256/tfrecord
DATASET_DIR=datasets/google/regular/tfrecord
# DATASET_DIR=datasets/mnist

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim

# A helper function for printing pretty output.
Banner () {
    local text=$1
    local green='\033[0;32m'
    local nc='\033[0m'  # No color.
    echo -e "${green}${text}${nc}"
}

Banner "Starting ${run_mode} for ${NUM_EPOCHS} epochs..."

# Run temporary tests.
if [[ "$run_mode" == "test" ]]; then
    ${PYTHON} "${src_dir}/data_provider_test.py" 
fi

# Run training.
if [[ "$run_mode" == "training" ]]; then
    ${PYTHON} "${src_dir}/train.py" \
        --train_log_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --mode="training" \
        --network="alexnet" \
        --optimizer="Adam" \
        --num_epochs=${NUM_EPOCHS} \
        --alsologtostderr
fi

# Run triplet training.
if [[ "$run_mode" == "triplet_training" ]]; then
    EPOCHS_PER_RUN=10
    NUM_RUN=$((NUM_EPOCHS / EPOCHS_PER_RUN))
    for ((i=1; i<=NUM_RUN; i++))
        do
        ${PYTHON} "${src_dir}/train.py" \
            --train_log_dir=${TRAIN_DIR} \
            --dataset_dir=${DATASET_DIR} \
            --mode="triplet_training" \
            --network="alexnet" \
            --optimizer="Adam" \
            --batch_size=512 \
            --learning_rate=1e-6 \
            --num_epochs=$((EPOCHS_PER_RUN * i)) \
            --alsologtostderr
        done
fi

# Run visualization
if [[ "$run_mode" == "visualization" ]]; then
    ${PYTHON} "${src_dir}/train.py" \
        --train_log_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --mode="visualization" \
        --alsologtostderr
fi

Banner "Finished ${run_mode} for ${NUM_EPOCHS} epochs."
