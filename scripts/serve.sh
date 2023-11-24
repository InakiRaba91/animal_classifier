#!/bin/bash
docker build -t animal_classifier:latest -f docker/Dockerfile.classifier .
docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 animal_classifier:latest