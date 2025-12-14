#!/bin/bash

# Configuration variables
MODEL_PATH="path_to_your_model"

################################################################################
# Configuration for discovering ODE system structures (Port 5000)
################################################################################
python ./llm_engine/engine.py \
  --model_path "$MODEL_PATH" \
  --gpu_ids 0  \
  --port 5000 \
  --quantization &

################################################################################
# Configuration for inferring parameter distributions (Port 5001)
################################################################################
python ./llm_engine/engine.py \
  --model_path "$MODEL_PATH" \
  --gpu_ids 0  \
  --port 5001 \
  --quantization &

wait
