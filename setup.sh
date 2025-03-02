#!/bin/bash
REMOTE_USER="root"
REMOTE_HOST="195.242.10.170"
REMOTE_PORT="1123"
REMOTE_DIR="/root/project"
HOME_DIR="/home"
VENV_DIR="${HOME_DIR}/venv"
SSH_CMD="ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"

echo "=== Starting Setup Process ==="

# Function to run command on remote and check status
run_remote() {
    $SSH_CMD "$1"
    return $?
}

echo "Step 1: Setting up remote environment..."
run_remote "mkdir -p ${REMOTE_DIR} ${HOME_DIR} && \
    apt-get update && \
    apt-get install -y rsync python3-dev"

echo "Step 2: Syncing project files..."
rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'gsm8k_train.json' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

echo "Step 3: Setting up Python environment..."
# Clear GPU memory from previous processes
run_remote "nvidia-smi --query-compute-apps=pid --format=csv | tail -n +2 | xargs -I {} kill -9 {} 2>/dev/null || true"

# Create venv, install dependencies including bitsandbytes
run_remote "if [ ! -d ${VENV_DIR} ]; then \
    apt-get update && apt-get install -y python3-venv && \
    python3 -m venv ${VENV_DIR}; \
fi && \
source ${VENV_DIR}/bin/activate && \
pip install -U pip && \
pip install -r ${REMOTE_DIR}/requirements.txt && \
pip install bitsandbytes>=0.43.3 && \
pip show bitsandbytes && \
python3 -c 'import torch; print(\"CUDA Available:\", torch.cuda.is_available())'"

echo "Step 4: Pre-caching model weights..."
run_remote "source ${VENV_DIR}/bin/activate && \
    python3 -c 'from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained(\"agentica-org/DeepScaleR-1.5B-Preview\")'"

echo "Step 5: Preparing dataset..."
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

echo "=== Setup Complete ==="
