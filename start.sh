#!/bin/bash

# Start Flask backend using Gunicorn in the background
# Your main.py is at the root, so reference it correctly as `main:app`.
echo "Starting Flask backend..."
gunicorn main:app --workers 1 --bind 0.0.0.0:5000 --timeout 600 --log-level debug &

# Wait a bit for Flask to start (optional, but can help prevent connection errors)
sleep 5

# Start Streamlit frontend in the foreground, listening on the public port (7860).
# Remove --server.enableCORS false if not strictly needed, but keep --server.enableXsrfProtection false for file uploads.
echo "Starting Streamlit frontend..."
streamlit run streamlit_app.py \
  --server.port 7860 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType none
# Note: Streamlit is run in the foreground so it becomes the primary process for the container.