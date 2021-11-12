# variables for artifacts
ARTIFACTS="confusion_matrix.jpg|dev_label_matrix.npy|development_data.html|development_data.pkl|lf_summary_dev.html|label_model|label_model.pkl|lf_summary_train.html|log.txt|train_label_matrix.npy|training_data.html|training_data.pkl"

echo "Testing multiclass..."
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e example --experiment-name test1 -P train_data=/unlabeled_data/multiclass_df.pkl -P dev_data=0 -P task=multiclass -P seed=8 ."
echo "Checking for all artifacts..."
docker exec dp-mlflow sh -c "if [[ $( ls ./mlruns/1/**/artifacts | grep -ic -E $ARTIFACTS ) != 7 ]]; then exit 1; fi"
echo "Checking for no errors or warnings..."
docker exec dp-mlflow sh -c "! grep -ni -e "WARNING" -e "ERROR" ./mlruns/1/**/artifacts/log.txt"
echo "☑ Multiclass test passed."

echo "Testing multiclass with development data..."
docker exec dp-mlflow mv /annotations/example/multiclass_gold_df.pkl /annotations/example/gold_df.pkl
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e example --experiment-name test2 -P train_data=/unlabeled_data/multiclass_df.pkl -P dev_data=1 -P task=multiclass -P seed=8 ."
docker exec dp-mlflow mv /annotations/example/gold_df.pkl /annotations/example/multiclass_gold_df.pkl
echo "Checking for all artifacts..."
docker exec dp-mlflow sh -c "if [[ $( ls ./mlruns/2/**/artifacts | grep -ic -E $ARTIFACTS ) != 12 ]]; then exit 1; fi"
echo "Checking for no errors or warnings..."
docker exec dp-mlflow sh -c "! grep -ni -e "WARNING" -e "ERROR" ./mlruns/2/**/artifacts/log.txt"
echo "☑ Multiclass with development data test passed."

echo "Testing multilabel..."
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e example --experiment-name test3 -P train_data=/unlabeled_data/multilabel_df.pkl -P dev_data=0 -P task=multilabel -P seed=8 ."
echo "Checking for all artifacts..."
docker exec dp-mlflow sh -c "if [[ $( ls ./mlruns/3/**/artifacts | grep -ic -E $ARTIFACTS ) != 7 ]]; then exit 1; fi"
echo "Checking for no errors or warnings..."
docker exec dp-mlflow sh -c "! grep -ni -e "WARNING" -e "ERROR" ./mlruns/3/**/artifacts/log.txt"
echo "☑ Multilabel test passed."

echo "Testing multilabel with development data..."
docker exec dp-mlflow mv /annotations/example/multilabel_gold_df.pkl /annotations/example/gold_df.pkl
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e example --experiment-name test4 -P train_data=/unlabeled_data/multilabel_df.pkl -P dev_data=1 -P task=multilabel -P seed=8 ."
docker exec dp-mlflow mv /annotations/example/gold_df.pkl /annotations/example/multilabel_gold_df.pkl
echo "Checking for all artifacts..."
docker exec dp-mlflow sh -c "if [[ $( ls ./mlruns/4/**/artifacts | grep -ic -E $ARTIFACTS ) != 11 ]]; then exit 1; fi"
echo "Checking for no errors or warnings..."
docker exec dp-mlflow sh -c "! grep -ni -e "WARNING" -e "ERROR" ./mlruns/4/**/artifacts/log.txt"
echo "☑ Multilabel with development data test passed."

#if [[ $(grep -ic -E $ARTIFACTS docker-compose.yml) = 110 ]]; then echo "yes"; else echo "no"; fi
#yes
#if [[ $(grep -ic -E $ARTIFACTS docker-compose.yml) = 100 ]]; then echo "yes"; else echo "no"; fi
#no
