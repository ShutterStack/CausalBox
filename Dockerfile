# Use a slim Python base image for efficiency
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker's build cache.
COPY requirements.txt .

# Install Python dependencies (including gunicorn, Flask, Streamlit, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container's /app directory.
COPY . .

# Grant execution permissions to the start.sh script
RUN chmod +x start.sh

# Expose the port Streamlit will listen on directly (7860 for Hugging Face Spaces)
EXPOSE 7860

# Define the command to run when the container starts.
CMD ["./start.sh"]
