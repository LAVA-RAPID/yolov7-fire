#!/bin/bash

# Name of your Docker image
IMAGE_NAME="fire-v1"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "Docker image built successfully."

# Run the Docker container
echo "Running Docker container..."
docker run -it --rm \
    -v "$(pwd)/data:/app/data" \
    $IMAGE_NAME

# Add any arguments your app needs after $IMAGE_NAME above
# For example: $IMAGE_NAME --input /app/data/input.mp4 --output /app/data/output

echo "Docker container stopped."