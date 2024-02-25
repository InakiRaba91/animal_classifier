#!/bin/bash
docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -v ./model_store:/source/model_store animal_classifier_torchserve:latest