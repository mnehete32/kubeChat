# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This assumes requirements.txt is at the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the train_test_split script into the container
COPY components/train_test_split/train_test_split.py .

# Command to run the train_test_split script. This will be overridden by Kubeflow Pipelines.
ENTRYPOINT ["python", "train_test_split.py"]