#!/bin/bash
# ssh -p 1123 root@195.242.10.170
# Set variables
REMOTE_USER="root"
REMOTE_HOST="195.242.10.170"
REMOTE_PORT="1123"
REMOTE_DIR="/root/project"
HOME_DIR="/home"
VENV_DIR="${HOME_DIR}/venv"
SSH_CMD="ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"

echo "=== Starting Deployment and Training Process ==="

# Function to run command on remote and check status
run_remote() {
    $SSH_CMD "$1"
    return $?
}

echo "Step 1: Setting up remote environment..."
# Create project directory and ensure /home exists
run_remote "mkdir -p ${REMOTE_DIR} ${HOME_DIR} && \
    if ! command -v rsync &> /dev/null; then \
        apt-get update && apt-get install -y rsync; \
    fi"

echo "Step 2: Syncing project files..."
# Sync all project files
rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'gsm8k_train.json' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

echo "Step 3: Setting up Python environment..."
# Create and activate venv in /home if it doesn't exist
run_remote "if [ ! -d ${VENV_DIR} ]; then \
    apt-get update && apt-get install -y python3-venv && \
    python3 -m venv ${VENV_DIR}; \
fi && \
source ${VENV_DIR}/bin/activate && \
pip install -r ${REMOTE_DIR}/requirements.txt && \
python3 -c 'import torch; print(\"CUDA Available:\", torch.cuda.is_available())'"

echo "Step 4: Preparing dataset..."
# Download and convert GSM8K dataset if needed
run_remote "cd ${REMOTE_DIR} && \
    source ${VENV_DIR}/bin/activate && \
    if [ ! -f gsm8k_train.json ]; then \
        echo 'Downloading GSM8K dataset...' && \
        wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl && \
        python3 -c '
import json
data = []
with open(\"train.jsonl\", \"r\") as f:
    for line in f:
        item = json.loads(line)
        data.append({\"question\": item[\"question\"], \"answer\": item[\"answer\"]})
with open(\"gsm8k_train.json\", \"w\") as f:
    json.dump(data, f)
print(\"Dataset converted successfully!\")
'
    else
        echo 'Dataset already exists, skipping download.'
    fi"

echo "Step 5: Starting training..."
# Run the training script with the venv
run_remote "cd ${REMOTE_DIR} && source ${VENV_DIR}/bin/activate && python3 train.py"

echo "=== Process Complete ==="
