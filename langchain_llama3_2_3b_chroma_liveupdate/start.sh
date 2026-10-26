#!/bin/bash

# Start Ollama in background
# We use 'serve' to start the engine manually
ollama serve > /dev/null 2>&1 &

# Wait for Ollama to wake up
echo "Waiting for Ollama..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

# Pull the model (use llama3.1 for better performance as discussed!)
echo "Loading model..."
#ollama pull llama3.1
ollama pull llama3.2:3b
ollama pull nomic-embed-text # Ensure the fast embedding model is there

echo "Starting Web Interface..."
python3 app.py
