#!/bin/bash

#!/bin/bash

IMAGE_NAME="image_search"
CONTAINER_TAG="0.0.1"
HOST_PORT=3000
CONTAINER_PORT=3000
DB_VOLUME_PATH="./db"
BUILD=true

if [ "$BUILD" = true ]; then
    echo "Building Docker image..."
    docker build -t ${IMAGE_NAME}:${CONTAINER_TAG} .
else 
    echo "Skipping Docker image build..."
fi

if [ "$(docker ps -q -a -f name=${IMAGE_NAME}_container)" ]; then
    echo "Stopping running container: ${IMAGE_NAME}_container"
    docker stop ${IMAGE_NAME}_container

    if [ $? -eq 0 ]; then
        echo "Removing container: ${IMAGE_NAME}_container"
        docker rm ${IMAGE_NAME}_container
    else
        echo "Failed to stop container: ${IMAGE_NAME}_container"
        exit 1
    fi
else
    echo "No running container named ${IMAGE_NAME}_container found."
fi

echo "Running Docker container..."
# Can add --env-file .env once we need an env file
docker run -d \
  --name ${IMAGE_NAME}_container \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  ${IMAGE_NAME}:${CONTAINER_TAG}

echo "Docker container is running on port ${HOST_PORT}."