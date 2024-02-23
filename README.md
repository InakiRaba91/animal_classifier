# Train your model
Once you have labeled your data, you can train your model. First set up the environment running
```bash
poetry install
```

Then, execute the following command:
```bash
poetry run python -m animal_classifier training
```

# Serve your model
First you need to package your model. To do so, execute the following command from `apps/animal_classifier`:
```bash
poetry run torch-model-archiver --model-name animal --version 1.0 --model-file animal_classifier/models/model.py --serialized-file ./models/cats_and_dogs/<model-name>.pth --handler animal_classifier/api/torchserve/handler.py --export-path ./model_store/
```
poetry run torch-model-archiver --model-name animal --version 1.0 --model-file animal_classifier/models/model.py --serialized-file ./models/cats_and_dogs/AnimalNet_20231124-150755.pth --handler animal_classifier/api/handler.py --export-path ./model_store/

Then, you can serve your  in a Docker container. Go back to the root directory and run:
```bash
./scripts/serve_torchserve.sh
```

You can check the model is running by executing the following command:
```bash
curl http://localhost:8081/models/animal
```

Finally, you can make predictions using the following command:
- Single request
    ```bash
    curl http://localhost:8080/predictions/animal -T ./data/cats_and_dogs/frames/1.png
    ```
- Concurrent requests for batch processing:
    ```bash
    curl -X POST http://localhost:8080/predictions/animal -T ./data/cats_and_dogs/frames/1.png & curl -X POST http://localhost:8080/predictions/animal -T ./data/cats_and_dogs/frames/2.png
    ```

# Serve your Web API
You can serve your  in a Docker container. Go back to the root directory and run:
```bash
./scripts/serve_fastapi.sh
```

You can check the model is running by executing the following command:
```bash
curl http://localhost:8081/models/animal
```

Finally, you can make predictions using the following command:
```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"base_64_str": "your_base64_string_here"}'
```

To update config
"http://localhost:8000/api/setconfigtraffic?val=75"