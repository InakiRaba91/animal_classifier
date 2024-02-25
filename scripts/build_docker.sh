#!/bin/bash
docker build -t animal_classifier_fastapi:latest -f docker/Dockerfile.torchserve .