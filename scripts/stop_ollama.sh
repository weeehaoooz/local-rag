#!/bin/bash

# Target all processes containing "ollama" in their name or command path
echo "Checking for running Ollama processes..."

# List processes before killing
ps aux | grep -i ollama | grep -v grep

# Kill all matching processes (f matches full command line, i is case-insensitive usually but not always supported with -f, depends on pkill version)
# On macOS pkill -f follows pattern matching
echo "Stopping all Ollama instances (Server, App, and Runners)..."
pkill -f "Ollama"
pkill -f "ollama"

# Confirm they are stopped
sleep 1
REMAINING=$(ps aux | grep -i ollama | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "SUCCESS: All Ollama processes have been stopped."
else
    echo "WARNING: There are still $REMAINING Ollama processes running."
    ps aux | grep -i ollama | grep -v grep
fi
