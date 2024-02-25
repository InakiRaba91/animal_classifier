#!/bin/bash

# Get the Dockerfile extension from the first argument
dockerfile_extension=$1

docker build -t animal_classifier_torchserve:latest -f docker/Dockerfile.${dockerfile_extension} .