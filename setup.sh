#!/bin/bash

echo "[+] Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv cmake build-essential libboost-all-dev libopenblas-dev liblapack-dev libx11-dev

echo "[+] Creating virtual environment..."
python3 -m venv face_env
source face_env/bin/activate

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Installing Python requirements..."
pip install -r requirements.txt

echo "[+] Done. To activate later, run:"
echo "source face_env/bin/activate"
