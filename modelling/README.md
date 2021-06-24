# CECN modelling

Repo for data validation, data transformation, modelling, model analysis, and model serving

## References
https://medium.com/analytics-vidhya/serving-ml-with-flask-tensorflow-serving-and-docker-compose-fe69a9c1e369
https://www.tensorflow.org/tfx/guide/serving
https://docs.docker.com/compose/gettingstarted/
https://docs.docker.com/engine/reference/builder/
https://linuxhint.com/beginners_guide_docker_compose/
https://www.tensorflow.org/tfx/serving/api_rest
https://www.kdnuggets.com/2020/07/building-rest-api-tensorflow-serving-part-1.html
https://github.com/deepopinion/domain-adapted-atsc

Don't use this, it doesn't work, just keeping it here for reference
```shell script
mlflow models serve --no-conda -m <runs:/my-run-id/model-path> -h 0.0.0.0 
```

To train an end model:
```shell script
docker attach modelling-mlflow
mlflow run --no-conda -e <mlp> --experiment-name <blogs> -P train_data=/train_data</path/to/train_data.pkl> -P test_data=/test_data</path/to/test_data.pkl> -P features=<features> .
```

Look at the MLproject file for a full list of hyperparameters and what the defaults for them are, also the individual
modules have all the details. This program will likely be buggy becuase I was in a naming transition that I didn't yet
complete (I will certainly do so when I get back.) If you manage to fix the bugs and get a model trained, then change
the PV_MODEL_PATH variable in docker-compose.yml to point to your new model (MlFlow will report the experiment id and
whatnot) and restart everything by going `docker-compose up -d`. You should be able to access the docs for the API at our
VM IP with port 9000. 
