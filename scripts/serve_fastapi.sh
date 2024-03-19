#!/bin/bash
docker run --rm -p 8000:8000 -v ./config:/source/config animal_classifier_fastapi:latest