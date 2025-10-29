#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 851725541946.dkr.ecr.ap-south-1.amazonaws.com

# Pull the latest image
docker pull 851725541946.dkr.ecr.ap-south-1.amazonaws.com/autonexus_ecr_image:latest

if [ "$(docker ps -q -f name=my-app)" ]; then
    # Stop the running container
    docker stop my-app
fi

# Check if the container 'campusx-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=my-app)" ]; then
    # Remove the container if it exists
    docker rm my-app
fi

# Run a new container
docker run -d -p 80:8000 -e DAGSHUB_PAT=755ef83398b125a97f96e22292e6879f103e0f27 --name my-app 851725541946.dkr.ecr.ap-south-1.amazonaws.com/autonexus_ecr_image:latest