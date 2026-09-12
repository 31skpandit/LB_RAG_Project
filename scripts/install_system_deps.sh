#!/usr/bin/env bash
# System-level dependencies for Unstructured PDF parsing + Redis docstore.
set -e

echo "Installing OCR/PDF system dependencies (tesseract, poppler)..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Plain redis-server (from Ubuntu's own repo) is enough here: retrieval/doc_store.py
# only uses it as a key-value byte store (langchain's RedisStore) for raw text/
# table/image content. Vector search is handled separately by Chroma, so none
# of Redis Stack's extra modules (RediSearch/RedisJSON) are needed.
echo "Installing Redis..."
sudo apt-get install -y redis-server

echo "Starting redis-server in the background..."
redis-server --daemonize yes

echo "Done."
