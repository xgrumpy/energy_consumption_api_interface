# Use the official Python image as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Ensure the main script is executable
RUN chmod +x /app/main.py

EXPOSE 8001

# Command to run your script
CMD ["python", "main.py"]
