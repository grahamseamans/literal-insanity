#!/bin/bash
REMOTE_USER="root"
REMOTE_HOST="195.242.10.170"
REMOTE_PORT="1123"
REMOTE_DIR="/root/project"
HOME_DIR="/home"
VENV_DIR="${HOME_DIR}/venv"
SSH_CMD="ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"

echo "=== Starting Training Process ==="

# Function to run command on remote and check status
run_remote() {
    $SSH_CMD "$1"
    return $?
}

echo "Running training..."
run_remote "cd ${REMOTE_DIR} && source ${VENV_DIR}/bin/activate && python3 train.py"

echo "=== Training Complete ==="
