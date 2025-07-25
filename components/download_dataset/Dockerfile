# Use a lightweight Python base image
FROM python:3.11.4-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This assumes requirements.txt is at the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY download_dataset.py .

ENTRYPOINT ["python", "download_dataset.py"]