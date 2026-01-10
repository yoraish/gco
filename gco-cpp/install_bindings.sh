#!/bin/bash

# GCo C++ Python Bindings Installation Script

set -e  # Exit on any error

echo "Installing GCo C++ Python Bindings..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the gco-cpp directory."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install the module in development mode
echo "Building and installing the module..."
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .

echo "[OK] Installation complete!"
