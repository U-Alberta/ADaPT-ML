# CECN classification

Repo for data validation, data transformation, modeling, model analysis, and model serving

## References
https://medium.com/analytics-vidhya/serving-ml-with-flask-tensorflow-serving-and-docker-compose-fe69a9c1e369
https://www.tensorflow.org/tfx/guide/serving
https://docs.docker.com/compose/gettingstarted/
https://docs.docker.com/engine/reference/builder/
https://linuxhint.com/beginners_guide_docker_compose/
https://www.tensorflow.org/tfx/serving/api_rest
https://www.kdnuggets.com/2020/07/building-rest-api-tensorflow-serving-part-1.html
https://github.com/deepopinion/domain-adapted-atsc

```bash script
mlflow models serve -m /Users/mlflow/mlflow-prototype/mlruns/0/7c1a0d5c42844dcdb8f5191146925174/artifacts/model -p 1234
```