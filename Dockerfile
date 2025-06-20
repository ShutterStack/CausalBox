# Use a Python base image with a good balance of size and features
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (Nginx, etc.)
# Run apt-get update first, then install packages, then clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy the combined requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Flask backend and Streamlit frontend code
COPY flask_backend/ ./flask_backend/
COPY streamlit_frontend/ ./streamlit_frontend/
# If you have a sample dataset you want to include
COPY data/ ./data/

# Copy Nginx configuration and startup script
COPY nginx.conf /etc/nginx/sites-available/default
COPY start.sh .

# Ensure Nginx uses our config
RUN ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default && \
    rm -rf /etc/nginx/sites-enabled/default.bak

# Make the startup script executable
RUN chmod +x start.sh

# Expose the port Nginx will listen on (Hugging Face Spaces will expose this to the internet)
EXPOSE 7860

# Command to run on container startup
CMD ["./start.sh"]