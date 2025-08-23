#!/bin/bash

echo "Japanese Video Subtitle Generator"
echo "================================"
echo ""
echo "Starting GUI..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Try to activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Virtual environment activated."
elif [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated."
else
    echo "No virtual environment found, using system Python."
fi

# Run the GUI
python3 gui.py

# If GUI fails, show error
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: GUI failed to start."
    echo "Please check that all dependencies are installed:"
    echo "pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo ""
echo "Press Enter to exit..."
read
