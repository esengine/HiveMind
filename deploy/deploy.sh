#!/bin/bash
# HiveMind Server Deployment Script
# For low-resource servers (2GB RAM)

set -e

echo "=========================================="
echo "HiveMind Server Deployment"
echo "=========================================="

# Configuration
INSTALL_DIR="/opt/hivemind"
DATA_DIR="/var/lib/hivemind"
USER="hivemind"
REPO_URL="https://github.com/esengine/HiveMind.git"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

echo "[1/7] Installing system dependencies..."
if command -v yum &> /dev/null; then
    # TencentOS / CentOS / RHEL
    yum install -y python3 python3-pip python3-venv git
elif command -v apt &> /dev/null; then
    # Ubuntu / Debian
    apt update
    apt install -y python3 python3-pip python3-venv git
else
    echo -e "${RED}Unsupported package manager${NC}"
    exit 1
fi

echo "[2/7] Creating hivemind user..."
if ! id "$USER" &>/dev/null; then
    useradd -r -s /bin/false $USER
fi

echo "[3/7] Creating directories..."
mkdir -p $INSTALL_DIR
mkdir -p $DATA_DIR/adapters
mkdir -p $DATA_DIR/logs

echo "[4/7] Cloning repository..."
if [ -d "$INSTALL_DIR/.git" ]; then
    cd $INSTALL_DIR
    git pull
else
    rm -rf $INSTALL_DIR/*
    git clone $REPO_URL $INSTALL_DIR
fi

echo "[5/7] Setting up Python environment..."
cd $INSTALL_DIR
python3 -m venv venv
source venv/bin/activate

# Install lightweight dependencies only
pip install --upgrade pip
pip install -r requirements-server.txt

echo "[6/7] Setting permissions..."
chown -R $USER:$USER $INSTALL_DIR
chown -R $USER:$USER $DATA_DIR

echo "[7/7] Installing systemd service..."
cp $INSTALL_DIR/deploy/hivemind.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable hivemind
systemctl restart hivemind

# Check status
sleep 2
if systemctl is-active --quiet hivemind; then
    echo -e "${GREEN}=========================================="
    echo "Deployment successful!"
    echo "=========================================="
    echo "Service status: running"
    echo "API endpoint: http://$(hostname -I | awk '{print $1}'):8000"
    echo "API docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
    echo ""
    echo "Commands:"
    echo "  Check status:  systemctl status hivemind"
    echo "  View logs:     journalctl -u hivemind -f"
    echo "  Restart:       systemctl restart hivemind"
    echo -e "==========================================${NC}"
else
    echo -e "${RED}=========================================="
    echo "Deployment failed!"
    echo "Check logs: journalctl -u hivemind -e"
    echo -e "==========================================${NC}"
    exit 1
fi
