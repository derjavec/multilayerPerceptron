#!/bin/bash

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Existing virtual environment found. Removing it..."
    rm -rf .venv
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

echo "Environment setup complete!"
echo "Activate it with: source .venv/bin/activate"
