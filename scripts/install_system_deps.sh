#!/usr/bin/env bash
# System-level dependencies for Unstructured PDF parsing + Redis docstore.
set -e

echo "Installing OCR/PDF system dependencies (tesseract, poppler)..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

echo "Installing Redis Stack Server..."
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install -y redis-stack-server

echo "Starting redis-stack-server in the background..."
redis-stack-server --daemonize yes

echo "Done."
