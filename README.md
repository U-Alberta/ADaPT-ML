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

```shell script
mlflow models serve --no-conda -m <runs:/my-run-id/model-path> -h 0.0.0.0 
```

```shell script
mlflow run --no-conda -e <mlogit> --experiment-name <energyeast> -P train_data=/train_data</path/to/train_data.pkl> -P test_data=/test_data</path/to/test_data.pkl> .
```
