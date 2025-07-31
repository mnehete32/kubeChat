# Use a lightweight Python base image
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This assumes requirements.txt is at the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the train script into the container
COPY . .

ENV LD_PRELOAD=/usr/local/lib/python3.10/dist-packages/scikit_learn.libs/libgomp-a34b3233.so.1.0.0

# Command to run the train script. This will be overridden by Kubeflow Pipelines.
ENTRYPOINT ["python", "train.py"]