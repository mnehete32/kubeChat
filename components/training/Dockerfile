# Use a lightweight Python base image
FROM python:3.11.4-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This assumes requirements.txt is at the project root
COPY components/training/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the train script into the container
COPY components/training/train.py .

ENV LD_PRELOAD=/usr/local/lib/python3.11/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# Command to run the train script. This will be overridden by Kubeflow Pipelines.
ENTRYPOINT ["python", "train.py"]