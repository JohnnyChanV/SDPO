#!/bin/bash

# Usage: ./run_local_opcd.sh [experiment_name_suffix]
#
# Few-shot Context Distillation (OPCD) training script.
# Teacher: frozen model with per-query dynamic few-shot examples
# Student: trainable model with zero-shot prompt

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="opcd"

# Dataset paths (modify for your dataset)
DATA_PATH="datasets/essay_comment"

# Model
MODEL_PATH="Qwen/Qwen3-4B"

# Hyperparameters
TRAIN_BATCH_SIZE=32
LR=1e-5
ALPHA=0.5                    # 0.0=forward KL, 0.5=JSD, 1.0=reverse KL
FEW_SHOT_K=30
FEW_SHOT_PLACEMENT="message" # "system" or "message"
TOTAL_STEPS=50
TEACHER_MODE="frozen"        # "frozen", "ema", "periodic"

# Few-shot data paths (MUST set these for your dataset)
FEW_SHOT_DEV_DATA=""         # e.g., data/essay_comment/dev_data.json
FEW_SHOT_DEV_MSG_DATA=""     # e.g., path/to/essay_sampling_reasoning.jsonl
TRAIN_DIST_MATRIX=""         # e.g., path/to/dev_x_dev/query_x_candidate.npy
EVAL_DIST_MATRIX=""          # e.g., path/to/test_x_dev/query_x_candidate.npy

export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_opcd"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="OPCD-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-k${FEW_SHOT_K}-lr${LR}-${TEACHER_MODE}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=OPCD-local \
trainer.total_training_steps=$TOTAL_STEPS \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.few_shot_k=$FEW_SHOT_K \
actor_rollout_ref.actor.self_distillation.few_shot_placement=$FEW_SHOT_PLACEMENT \
actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_MODE \
actor_rollout_ref.actor.self_distillation.distillation_topk=100"

# Add few-shot data paths if set
if [ -n "$FEW_SHOT_DEV_DATA" ]; then
    ARGS="$ARGS actor_rollout_ref.actor.self_distillation.few_shot_dev_data=$FEW_SHOT_DEV_DATA"
fi
if [ -n "$FEW_SHOT_DEV_MSG_DATA" ]; then
    ARGS="$ARGS actor_rollout_ref.actor.self_distillation.few_shot_dev_msg_data=$FEW_SHOT_DEV_MSG_DATA"
fi
if [ -n "$TRAIN_DIST_MATRIX" ]; then
    ARGS="$ARGS actor_rollout_ref.actor.self_distillation.few_shot_train_distance_matrix=$TRAIN_DIST_MATRIX"
fi
if [ -n "$EVAL_DIST_MATRIX" ]; then
    ARGS="$ARGS actor_rollout_ref.actor.self_distillation.few_shot_eval_distance_matrix=$EVAL_DIST_MATRIX"
fi

echo "================================================================"
echo "Starting OPCD (Few-shot Context Distillation) Training"
echo "================================================================"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Alpha: $ALPHA (0=fwd KL, 0.5=JSD, 1=rev KL)"
echo "Few-shot K: $FEW_SHOT_K"
echo "Placement: $FEW_SHOT_PLACEMENT"
echo "Teacher mode: $TEACHER_MODE"
echo "Steps: $TOTAL_STEPS"
echo "================================================================"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
