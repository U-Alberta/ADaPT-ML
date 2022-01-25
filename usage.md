## Installation and Usage Guidelines ##

Follow these guidelines to set up ADaPT-ML on your machine and to see how you can add new classification tasks to the system. Each header links to the appropriate file created for the [Example Use Case](./README.md) so you can see an example of these instructions implemented.

### Step 1: Review System Requirements ###

#### Required Setup ####
The following are required to run ADaPT-ML:
- [Docker Engine v19.03.0+](https://docs.docker.com/)
- [Docker Compose v1 1.29.2](https://docs.docker.com/compose/) (This software has not been tested with a newer version of Docker Compose)
- \>20 GB in your Docker root directory (usually /var/lib/docker) for storing images

#### Recommended Setup ####
This system configuration has been tested, and will get you up and running with ADaPT-ML faster.
- Linux or MacOS (The automated tests on GitHub use Ubuntu 20.04.3 LTS)
- Perform these CrateDB [bootstrap checks](https://crate.io/docs/crate/howtos/en/latest/admin/bootstrap-checks.html), as the host system must be configured correctly to use CrateDB with Docker.

### Step 2: [Set up the environment variables for Docker Compose](./.env) ###
**It is recommended that at this point, you test ADaPT-ML by following these [instructions](./test/testing.md)**. After testing, make a copy of the `.env` file in the repository's root directory and call it `.env.dev`. Review the `.env.dev` file, and edit the variables according to their descriptions.

### Step 3: Changes to [label-studio](./label-studio) ###

Most of the setup for Label Studio is done through the UI that is launched at http://localhost:8080 by default, but there are a few things within this project directory to take note of, especially if you plan on using Label Studio's API.

#### (a) [Format your Labeling Config file](./label-studio/config/example_config.xml) ####

This configures how each component of a datapoint will be displayed to the annotators. This file can be copied and pasted into the Label Studio Labeling Configuration UI, or set for a certain project [using the API](https://labelstud.io/api#operation/api_projects_create).

#### (b) [Define your classification task name and classes](./label-studio/ls/__init__.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), you will need to update the `CLASSIFICATION_TASKS` variable with your new task name and corresponding classes.

#### Please note: ####

The Example Use Case demonstrates how to add a new classification task with only a text component for each datapoint. Therefore, it may be necessary to make changes to the [task sampling](./label-studio/ls/sample_tasks.py), [annotation processing](./label-studio/ls/process_annotations.py), and/or [annotator agreement](./label-studio/ls/annotator_agreement.py) modules if Label Studio's JSON import and export format is different according to the datapoint's number of components (e.g. both text and image), number of annotators, etc. See [Contributing](#community-guidelines).

### Step 4: Changes to [data-programming](./data-programming) ###

Setting up the data-programming project within ADaPT-ML to work with a new classification task requires adding new Python modules and editing some existing files.

#### (a) [Define your class names](./data-programming/label/lfs/__init__.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), this is where you need to define the Class that will hold both the name of each class and the number representing that class, which the Labeling Functions will use to vote, and which will ultimately make up the Label Matrix. **NOTE**: if your task is specifically a **binary** task, then you need to use the suffix `_pos` for the positive class (and optionally `_neg` for the negative class) in order to have the correct binary classification metrics downstream.

#### (b) [Write your Labeling Functions](./data-programming/label/lfs/example.py) ####

Create a module within [./data-programming/label/lfs](data-programming/label/lfs). This module that you can name after your new classification task is where you will write your Labeling Functions, and create a function called `get_lfs` that will produce an iterable containing all of the Labeling Functions you have defined.

#### (c) [Create your main function as an MLflow endpoint](./data-programming/label/example.py) ####

Create a module within [./data-programming/label](./data-programming/label). This is the main module for your new task. You will need to import the Class you defined in [Step 4(a)](#a-define-your-class-namesdata-programminglabellfs__init__py) and the `get_lfs` function defined in [Step 4(b)](#b-write-your-labeling-functionsdata-programminglabellfsexamplepy). You will also need to create a name for the Label Model that will be specific to your new task, and a dictionary with the names of the columns holding the features extracted for use with the Labeling Functions you defined as keys and any functions necessary to properly transform or unpack the featurized data point as values. You will also need to specify the path within the Label Studio annotations directory to the DataFrame that holds the annotated development data. Here you can add additional arguments to the argument parser if your Labeling Functions need them, like thresholds.

#### (d) [Add your MLflow endpoint to the MLproject file](./data-programming/MLproject) ####

This file is where you will specify the default hyperparameters for training the Label Model, additional parameters for your Labeling Functions, the type of classification your new task falls under (multiclass or multilabel), and the path to the main module you created in [Step 4(c)](#c-create-your-main-function-as-an-mlflow-endpointdata-programminglabelexamplepy). If you perform hyperparameter tuning and find a configuration that works well for your task, then change the defaults here!

### Step 5: Changes to [modelling](./modelling) (including model deployment) ###

There is not much that you have to edit in this project directory unless you need a machine learning algorithm other than a multi-layer perceptron (MLP), but if you do add a new algorithm, please see [Contributing](#community-guidelines)! For now, all of the edits are to the FastAPI app.

#### (a) [Add your class response format and endpoint to the deployment app](./modelling/app/main.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), you will need to add a response model that validates the output from your prediction endpoint. You will also need to create and set environment variables in [Step 2](#step-2-set-up-the-environment-variables-for-docker-composeenv) for your new End Model and add functions to load them. You can add an element to the `loaded_models_dict` for your model, so you will know if it loaded successfully by visiting the root page. Finally, you will need to add an endpoint to get predictions for new datapoints from your model. This endpoint can return a JSON response in the format of your specified response model, or directly update the data in CrateDB with the predictions. 

### Step 6: Start ADaPT-ML ###

Once you have your new classification task ready to go by completing Steps 1-5, all you need to do is:
```shell
cd ADaPT-ML/
docker-compose --env-file .env.dev --profile dev up -d
docker-compose ps
```
Once you see Docker Compose report this:
```
         Name                        Command                  State                                  Ports                            
--------------------------------------------------------------------------------------------------------------------------------------
crate-db                  /docker-entrypoint.sh crat ...   Up             0.0.0.0:4200->4200/tcp,:::4200->4200/tcp, 4300/tcp, 5432/tcp
dp-mlflow                 /bin/bash                        Up                                                                         
dp-mlflow-db              /entrypoint.sh mysqld            Up (healthy)   3306/tcp, 33060/tcp, 33061/tcp                              
dp-mlflow-server          mlflow server --backend-st ...   Up             0.0.0.0:5000->5000/tcp,:::5000->5000/tcp                    
label-studio-dev          /bin/bash                        Up                                                                         
label-studio-web          ./deploy/docker-entrypoint ...   Up             0.0.0.0:8080->8080/tcp,:::8080->8080/tcp                    
modelling-mlflow          /bin/bash                        Up                                                                         
modelling-mlflow-db       /entrypoint.sh mysqld            Up (healthy)   3306/tcp, 33060/tcp, 33061/tcp                              
modelling-mlflow-deploy   /start.sh                        Up             0.0.0.0:80->80/tcp,:::80->80/tcp                            
modelling-mlflow-server   mlflow server --backend-st ...   Up             0.0.0.0:5001->5000/tcp,:::5001->5000/tcp 
```
Then it's ready! Import your data into a table in CrateDB and refer to the [Example Usage](./README.md) for an example of how to manipulate the data so that it's ready for ADaPT-ML. How you load the data, featurize it, and sample from it to create your unlabeled training data is up to you -- ADaPT-ML does not perform these tasks. However, there may be an opportunity for certain sampling methods to become a part of the system; see [Contributing](#community-guidelines).

### Optional: Create a dev/test dataset using Label Studio ###

If you have two or more domain experts available to label some datapoints in order to create a gold dev/test dataset, then you can follow these steps to use Label Studio to accomplish this.

#### (a) Sample some data from CrateDB ####

Use this module to sample N random datapoints from a table in CrateDB, making sure to include the columns that contain the data that the domain exports will use during annotation.
```shell
docker exec label-studio-dev python ./ls/sample_tasks.py --help
```
```
usage: sample_tasks.py [-h] [--filename FILENAME] table columns [columns ...] n {example}

Sample a number of data points from a table to annotate.

positional arguments:
  table                Table name that stores the data points.
  columns              column name(s) of the data point fields to use for annotation.
  n                    Number of data points to sample.
  {example}            What classification task is this sample for?

optional arguments:
  -h, --help           show this help message and exit
  --filename FILENAME  What would you like the task file to be called?
```

#### (b) Use Label Studio's UI or API to label the sampled datapoints ####

Please refer to these guides to create an account ([UI](https://labelstud.io/guide/signup.html#Create-an-account) or [API](https://labelstud.io/api#operation/api_users_create)), create a project ([UI](https://labelstud.io/guide/setup_project.html#Create-a-project) or [API](https://labelstud.io/api#operation/api_projects_create)), load the tasks file created in [Optional (a)](#optional-create-a-devtest-dataset-using-label-studio) ([UI](https://labelstud.io/guide/tasks.html#Import-data-from-the-Label-Studio-UI) or [API](https://labelstud.io/guide/tasks.html#Import-data-using-the-API)), label the tasks ([UI](https://labelstud.io/guide/labeling.html) or [API](https://labelstud.io/api#tag/Tasks)), and export the resulting annotations ([UI](https://labelstud.io/guide/export.html#Export-using-the-UI-in-Label-Studio) or [API](https://labelstud.io/guide/export.html#Export-using-the-API)).

#### (c) Process the exported annotations ####

Once you have exported the annotations and moved the file to `${LS_ANNOTATIONS_PATH}`, the annotations need to be processed using this module:
```shell
docker exec label-studio-dev python ./ls/process_annotations.py --help
```
```
usage: process_annotations.py [-h] filename {example} gold_choice

Format exported annotations into DataFrames ready for downstream functions.

positional arguments:
  filename     Name of the exported annotations file.
  {example}    Which task is the annotations file for?
  gold_choice  How to settle disagreements between workers. id: Provide the id of the worker whose labels will be chosen every time. random: The least strict. Choose the label that the majority of workers agree on. If they are evenly split, choose a worker label randomly. majority: More strict. Choose the
               label that the majority of workers agree on. If they are evenly split, drop that datapoint. drop: The most strict. If workers disagree at all, drop that datapoint.

optional arguments:
  -h, --help   show this help message and exit
```
This will save three DataFrames in `${LS_ANNOTATIONS_PATH}/[CLASSIFICATON_TASK]`, where `CLASSIFICATION_TASK` is the name of your new classification task defined in [Step 3(b)](#b-define-your-classification-task-name-and-classeslabel-studiols__init__py):
- `ann_df.pkl` contains all of the datapoints that were initially exported from Label Studio with a column for each annotator's label set.
- `task_df.pkl` contains only the datapoints that were labeled by all annotators working on the project (e.g., if worker 1 labeled 50 datapoints and worker 2 labeled 45, then this DataFrame will contain 45 datapoints.)
- `gold_df.pkl` contains the final gold label set that was compiled according to the method selected using the `gold_choice` argument.

#### (d) Calculate interannotator agreement ####

Before moving on with the gold labels in `gold_df.pkl`, this module should be used to determine the level of agreement between all of the annotators:
```shell
docker exec label-studio-dev ./ls/annotator_agreement.py --help
```
```
usage: annotator_agreement.py [-h] {example}

Compute the inter-annotator agreement for completed annotations.

positional arguments:
  {example}   Task to calculate agreement for.

optional arguments:
  -h, --help  show this help message and exit
```
This will log and print a report using Krippendorff's alpha.

### Step 7: Run a data programming experiment ###

Once you have determined how you will sample some of your data for training an End Model, you need to save it as a pickled Pandas DataFrame with columns `id` and `table_name`, and optionally other columns if you need them. `table_name` needs to have the name of the table in CrateDB where the datapoint is stored. Once this DataFrame is in the directory `$DP_DATA_PATH/unlabeled_data`, you can run this command to label your data:
```shell
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e [ENTRYPOINT] --experiment-name [EXP_NAME] -P train_data=/unlabeled_data/[DATA] -P dev_data=[0,1] -P task=[TASK] ."
```
where `ENTRYPOINT` is the name of the entrypoint you specified in [Step 4(d)](#d-add-your-mlflow-endpoint-to-the-mlproject-filedata-programmingmlproject), `EXP_NAME` is a name for the experiment of your choosing, `DATA` is the name of the pickled Pandas DataFrame holding your unlabeled data, `[0, 1]` is the flag to set indicating that you have done [Optional (c)](#c-process-the-exported-annotations) to create the `${LS_ANNOTATIONS_PATH}/[CLASSIFICATION_TASK]/gold_df.pkl`for your classification task (`1`), or that you do not have a gold dataset available from Label Studio (`0`), and `TASK` is the type of classification that is appropriate for your new task (multiclass or multilabel). You can then check http://localhost:5000 to access the MLflow UI and see the experiment log, Labeling Function evaluation, artifacts, metrics, and more. Your labeled data will be stored in the directory `${DP_DATA_PATH}/mlruns/EXP_ID/RUN_ID/artifacts/training_data.pkl` where `EXP_ID` is the id corresponding to `EXP_NAME`, and `RUN_ID` is a unique id created by MLflow for the run.

### Step 8: Run a modelling experiment ###

Once you have run some experiments and are happy with the resulting labeled data, take note of the `EXP_ID` and `RUN_ID` from [Step 7](#step-7-run-a-data-programming-experiment) within the filepath to the `training_data.pkl` and `development_data.pkl` (or, if you don't have a gold dataset from Label Studio, then instead of `development_data.pkl` you will use the DataFrame you split off of `training_data.pkl` and saved in the `artifacts` folder). Then you can run this command to train and evaluate an MLP model:
```shell
docker exec modelling-mlflow sh -c ". ~/.bashrc && wait-for-it modelling-mlflow-db:3306 -s -- mlflow run --no-conda -e mlp --experiment-name [EXP_NAME] -P train_data=/dp_mlruns/[EXP_ID]/[RUN_ID]/artifacts/training_data.pkl -P test_data=/dp_mlruns/[EXP_ID]/[RUN_ID]/artifacts/[TEST_DATA] -P features=FEATURE ."
```
where `EXP_NAME` is a name for the experiment of your choosing, `EXP_ID` and `RUN_ID` are from your evaluation from [Step 7](#step-7-run-a-data-programming-experiment), `TEST_DATA` is either `development_data.pkl` or the name of the Pandas DataFrame holding your testing data split off of `training_data.pkl`, and `FEATURE` is a list of column names holding the feature vectors in CrateDB. You can then check http://localhost:5001 to access the MLflow UI and see the experiment log, artifacts, metrics, and more.

### Step 9: Deploying your model ###

After you are satisfied with the performance of an End Model created in [Step 8](#step-8-run-a-modelling-experiment), take note of the `EXP_ID` and `RUN_ID` for the End Model, and update your End Model's [environment variable](#step-2-set-up-the-environment-variables-for-docker-composeenv) to `/mlruns/[EXP_ID]/[RUN_ID]/artifacts/mlp/python_model.pkl`. Then, edit the `environment` section of the `m_deploy` service in [docker-compose.yml](./docker-compose.yml) so that it has your End Model's environment variable.

Now you can reload the deployment API by running these commands:
```shell
docker-compose stop
docker-compose --profile dev up -d
```
and visit http://localhost:80/docs to see the deployment API. You can use this API to get predictions on unseen datapoints, and take note of the `curl` command to get predictions. It should look something like this:
```shell
curl -X 'POST' \
  'http://localhost/[ENDPOINT]' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "table_name": [
    [TABLE_NAME]
  ],
  "id": [
    [ID]
  ]
}'
```
where `ENDPOINT` is the one you created in [Step 5(a)](#step-5-changes-to-modellingmodelling-including-model-deployment), `TABLE_NAME` is a list of names of the table(s) containing the datapoints you need predictions for, and `ID` is the list of ids for the datapoints.


You now have predicted labels for your data and can perform any downstream analyses you need!

## Additional Installation Notes: ##

Check this out if you are hosting CrateDB or another SQLAlchemy-based database on a remote server:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html

If you want to train the Label Model using CUDA tensors, then please refer to these resources:
https://developer.nvidia.com/cuda-toolkit
https://pytorch.org/docs/stable/cuda.html

## Community Guidelines ##

### Contribution ###
Follow these guidelines to see where you can contribute to expand the system's functionality and adaptability. The following items are on ADaPT-ML's "wish list":
- a configuration file that can be used by the label-studio, data-programming, and modelling projects to automatically create the classification task directory for label studio, a coding schema for annotators, the Enum object that stores values that the Labeling Functions use, the ModelResponse schema for deployment, and anything else where it is important to have consistency and maintainability in the classification task name and classes.
- a main UI with links to all of the different UIs, buttons that can run commands to sample data and run end-to-end experiments by returning the `EXP_ID` and `RUN_ID` within mlruns for a successful and performant Label Model and End Model, forms for submitting new classification tasks, an interface that makes writing labeling functions easier, etc.
- implement some algorithms that can take a representative sample of a table in CrateDB for training data creation.
- implement classification algorithms in addition to the MLP.
- determine the best method for updating the CrateDB tables with worker labels, gold labels, Label Model labels and probabilities, and End Model predictions and probabilities.
- a separate project for creating a flexible feature store.

Please open an issue if you would like to propose an approach to adding these features.

### Report Issues and Seek Support ###
If you find a problem with the software or if you need help with any of the steps in this document or the testing document, please open an issue and I will try to address your concerns.
