# Use a lightweight Python base image
FROM python:3.11.4-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This assumes requirements.txt is at the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the EDA script into the container
COPY components/eda/eda.py .

# Command to run the EDA script. This will be overridden by Kubeflow Pipelines.
ENTRYPOINT ["python", "eda.py"]
