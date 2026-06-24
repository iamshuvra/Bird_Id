#!/usr/bin/env bash
# check local environment and active it

if [ ! -d "venv" ]; then
  echo "ERROR: venv not found"
  exit 1
fi

source venv/bin/activate


python3 inference.py
