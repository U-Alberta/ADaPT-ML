# CECN Data Programming

A dynamic approach to creating classifiers where the training data the model is built upon and the research questions
needing answers are always changing.

## References
https://medium.com/analytics-vidhya/serving-ml-with-flask-tensorflow-serving-and-docker-compose-fe69a9c1e369
https://www.tensorflow.org/tfx/guide/serving
https://docs.docker.com/compose/gettingstarted/
https://docs.docker.com/engine/reference/builder/

## First Time Setup

```shell script
make init
```

Will create the `venv`, install the requirements, install nltk models.

## Main Usage

### Step 1: Create the label matrix
#### Editing keyword LFs
There is a file `keywords.yaml` which can be edited to change the keywords associated with each class.

#### Creating the matrix using unlabeled training data
```shell script
python -m label --matrix
```
Creates a Label Matrix from the data in `unlabeled_train.csv` and saves it to `L_train.npy`

For Twitter, the data is expected to be in CSV format, headers included. The columns should be the tweet's `id_str`, `lang`, and some text. 
The text could be the tweet's `user_entered_text`, `quote_or_rt_text`, `poll_options`, `all_text`, `bio`, `hashtags`, 
or something else that might be useful. The different types of text will be in their own columns.

E.g.:
```
id_str,lang,all_text,hashtags\n
00001234,en,This is my tweet about energy! #canadarocks,#canadarocks\n
```

### Step 2: Train the label model
```shell script
python -m label --model
```
Takes the Label Matrix in `L_train.npy` and trains a label model. Saves it to `label_model.pkl`

### Step 3: Apply the label model to label the training data
```shell script
python -m label --apply
```
Takes the Label Model and applies it on the Label Matrix to get predictions and probabilities for each data point. Saves
the labeled training data to `labeled_train.csv`.

### Step 4: Train a classifier
```shell script
python -m classify --mlogit
```
Takes the training data and trains a specified classifier. Saves it to `classifier.pkl`

## Extra Functionality
### Run a small test of the whole system
```shell script
python -m test
```
