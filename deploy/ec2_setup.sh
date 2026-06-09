#!/bin/bash
# ============================================================
# EC2 Setup Script for LSTM Stock Predictor
# Run this on a fresh Ubuntu 22.04 EC2 instance
# Usage: bash ec2_setup.sh
# ============================================================

set -e  # Exit immediately if a command fails

echo "========================================"
echo "  LSTM Stock Predictor - EC2 Setup"
echo "========================================"

# ---- 1. System Update ----
echo "[1/7] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y

# ---- 2. Install Docker ----
echo "[2/7] Installing Docker..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ---- 3. Install Docker Compose (standalone) ----
echo "[3/7] Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# ---- 4. Add current user to docker group (no sudo needed) ----
echo "[4/7] Configuring Docker permissions..."
sudo usermod -aG docker $USER
newgrp docker || true

# ---- 5. Install Git ----
echo "[5/7] Installing Git..."
sudo apt-get install -y git

# ---- 6. Clone the Repository ----
echo "[6/7] Cloning repository..."
REPO_URL="https://github.com/Rudraksh-Rana/Stock-Predicter.git"
APP_DIR="$HOME/Stock-Predicter"

if [ -d "$APP_DIR" ]; then
    echo "  Repo already exists. Pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ---- 7. Create .env from example ----
echo "[7/7] Setting up environment..."
cd "$APP_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ⚠️  .env file created from .env.example"
    echo "  Please edit it with your values: nano .env"
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env:          nano $APP_DIR/.env"
echo "  2. Start services:     cd $APP_DIR && docker-compose up -d --build"
echo "  3. View logs:          docker-compose logs -f"
echo ""
echo "  FastAPI  → http://<YOUR-EC2-IP>:8000"
echo "  Streamlit→ http://<YOUR-EC2-IP>:8501"
echo ""
