# Use a PyTorch base image with CUDA support (if needed)
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the repository into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables if needed
ENV MODEL_NAME=VIZMed

# Default command to run inside the container
CMD ["python", "train.py"]
