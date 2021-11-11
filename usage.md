## Installation and Usage Guidelines ##

Follow these guidelines to set up ADaPT-ML on your machine and to see how you can add new classification tasks to the system.

### Step 1: Install Docker and Docker Compose ###
Docker and Docker Compose are required to use ADaPT-ML. Please follow the links for each and review the installation instructions, and make sure that they are installed on the host machine where you will be cloning or forking this repository to.

**If you want to test ADaPT-ML using the example use case at this point, proceed directly to [Step 6](#step-6-starting-adapt-ml) to get ADaPT-ML started, then to [Step 2](#step-2-create-a-gold-dataset-using-label-studio) of the [Example Use Case](#example-usage) to follow along. If you want to skip that and get started on adding your classification task to ADaPT-ML, then continue to Step 2 below.**

### Step 2: Set up the environment variables for Docker Compose ###
Review the `.env` file in the repository's root directory, and edit the variables according to their descriptions.

### Step 3: Changes to [label-studio](./label-studio) ###

Most of the setup for Label Studio is done through the UI that is launched at http://localhost:8080 by default when you run either the `dev` or `label` profile for docker compose, but there are a few things within this project directory to take note of, especially if you plan on using Label Studio's API.

#### [Format your Labeling Config file](./label-studio/config/example_config.xml) ####

This configures how each component of a datapoint will be displayed to the annotators. This file can be copied and pasted into the Label Studio Labeling Configuration UI, or set for a certain project [using the API](https://labelstud.io/api#operation/api_projects_create).

#### [Define your classification task name and classes](./label-studio/ls/__init__.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), you will need to update the `CLASSIFICATION_TASKS` variable with your new task name and corresponding classes.

#### Please note: ####

The Example Use Case demonstrates how to add a new classification task with only a text component for each datapoint. Therefore, it may be necessary to make changes to the [task sampling](./label-studio/ls/sample_tasks.py), [annotation processing](./label-studio/ls/process_annotations.py), and/or [annotator agreement](./label-studio/ls/annotator_agreement.py) modules if Label Studio's JSON import and export format is different according to the datapoint's number of components (e.g. both text and image), number of annotators, etc. See [Contributing](#community-guidelines).

### Step 4: Changes to [data-programming](./data-programming) ###

Setting up the data-programming project within ADaPT-ML to work with a new classification task requires adding new Python modules and editing some existing files. You can follow along with the Example Use Case to get an overview, but the details are as follows, file-by-file:

#### [Define your class names](./data-programming/label/lfs/__init__.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), this is where you need to define the Class that will hold both the name of each class and the number representing that class, which the Labeling Functions will use to vote, and which will ultimately make up the Label Matrix. **NOTE**: if your task is specifically a **binary** task, then you need to use the suffix `_pos` for the positive class (and optionally `_neg` for the negative class) in order to have the correct binary classification metrics downstream.

#### [Write your Labeling Functions](./data-programming/label/lfs/example.py) ####

This module that you can name after your new classification task is where you will write your Labeling Functions, and create a function called `get_lfs` that will produce an iterable containing all of the Labeling Functions you have defined.

#### [Create your main function as an MLflow endpoint](./data-programming/label/example.py) ####

This is the main module for your new task. You will need to import the Class you defined in [this step](#define-your-class-namesdata-programminglabellfs__init__py) and the `get_lfs` function defined in [this step](#write-your-labeling-functionsdata-programminglabellfsexamplepy). You will also need to create a name for the Label Model that will be specific to your new task, and a dictionary with the names of the columns holding the features extracted for use with the Labeling Functions you defined as keys and any functions necessary to properly transform or unpack the featurized data point as values. You will also need to specify the path within the Label Studio annotations directory to the DataFrame that holds the annotated development data. Here you can add additional arguments to the argument parser if your Labeling Functions need them, like thresholds.

#### [Add your MLflow endpoint to the MLproject file](./data-programming/MLproject) ####

This file is where you will specify the default hyperparameters for training the Label Model, additional parameters for your Labeling Functions, the type of classification your new task falls under (multiclass or multilabel), and the path to the main module you created in [this step](#create-your-main-function-as-an-mlflow-endpointdata-programminglabelexamplepy). If you perform hyperparameter tuning and find a configuration that works well for your task, then change the defaults here!

### Step 5: Changes to [modelling](./modelling) (including model deployment) ###

There is not much that you have to edit in this project directory unless you need a machine learning algorithm other than a multi-layer perceptron (MLP), but if you do add a new algorithm, please see [Contributing](#community-guidelines)! For now, all of the edits are to the FastAPI app.

#### [Add your class response format and endpoint to the deployment app](./modelling/app/main.py) ####

Until there is one configuration file for defining the classification task name and classes across all steps in the ADaPT-ML pipeline (see [Contributing](#community-guidelines)), you will need to add a response model that validates the output from your prediction endpoint. You will also need to create and set environment variables in [this step](#step-2-set-up-the-environment-variables-for-docker-compose) for your new End Model and add functions to load them. You can add an element to the `loaded_models_dict` for your model, so you will know if it loaded successfully by visiting the root page. Finally, you will need to add an endpoint to get predictions for new datapoints from your model.

### Step 6: Start ADaPT-ML ###

Once you have your new classification task ready to go by completing Steps 1-5, all you need to do is:
```shell
cd ADaPT-ML/
docker-compose --profile dev up -d
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
Then it's ready! Import your data into a table in CrateDB and refer to the [Example Usage](#example-usage) for an example of how to manipulate the data so that it's ready for ADaPT-ML. How you load the data, featurize it, and sample from it to create your unlabeled training data is up to you -- ADaPT-ML does not perform these tasks. However, there may be an opportunity for certain sampling methods to become a part of the system; see [Contributing](#community-guidelines).

### Step 7: Run a data programming experiment ###

Once you have determined how you will sample some of your data for training an End Model, you need to save it as a pickled Pandas DataFrame with columns `id` and `table_name`, and optionally other columns if you need them. `table_name` needs to have the name of the table in CrateDB where the datapoint is stored. Once this DataFrame is in the directory `$DP_DATA_PATH/unlabeled_data`, you can run this command to label your data:
```shell
docker exec dp-mlflow sh -c ". ~/.bashrc && wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e ENTRYPOINT --experiment-name EXP_NAME -P train_data=/unlabeled_data/DATA.pkl -P task=TASK ."
```
where `ENTRYPOINT` is the name of the entrypoint you specified in [this step](#add-your-mlflow-endpoint-to-the-mlproject-filedata-programmingmlproject), `EXP_NAME` is a name for the experiment of your choosing, `DATA` is the name of the Pandas DataFrame holding your unlabeled data, and `TASK` is the type of classification that is appropriate for your new task (multiclass or multilabel). You can then check http://localhost:5000 to access the MLflow UI and see the experiment log, Labeling Function evaluation, artifacts, metrics, and more. Your labeled data will be stored in the directory `$DP_DATA_PATH/mlruns/EXP_ID/RUN_ID/artifacts/training_data.pkl` where `EXP_ID` is the id corresponding to `EXP_NAME`, and `RUN_ID` is a unique id created by MLflow for the run.

### Step 8: Run a modelling experiment ###

Once you have run some experiments and are happy with the resulting labeled data, you can split this data into training and testing datasets according to a method of your choosing. You have probably noticed that up until this point, Label Studio has not actually been used; if you do have annotators who can manually label a sample of data, then you may choose to use this sample as your test set. Otherwise, if you want to externally validate the model later, then you can use the data that Label Studio labeled as your test data. Whatever you choose, make sure that you have placed `training_data.pkl` in `$MODELLING_DATA_PATH/train_data/` and your test data in `$MODELLING_DATA_PATH/test_data/`. Then you can run this command to train and evaluate an MLP model:
```shell
docker exec modelling-mlflow sh -c ". ~/.bashrc && wait-for-it modelling-mlflow-db:3306 -s -- mlflow run --no-conda -e mlp --experiment-name EXP_NAME -P train_data=/train_data/TRAIN_DATA.pkl -P test_data=/test_data/TEST_DATA.pkl -P features=FEATURE ."
```
where `EXP_NAME` is a name for the experiment of your choosing, `TRAIN_DATA` is the name of the Pandas DataFrame holding your training data, `TEST_DATA` is the name of the Pandas DataFrame holding your testing data, and `FEATURE` is a list of column names holding the feature vectors in CrateDB. You can then check http://localhost:5001 to access the MLflow UI and see the experiment log, artifacts, metrics, and more. Take note of the path to your trained End Model, and if you are satisfied with its performance, you can update your End Model's [environment variable](#step-2-setting-up-the-environment-variables-for-docker-compose) to this path.

### Step 9: Deploying your model ###

Edit the `environment` section of the `m_deploy` service in [docker-compose.yml](./docker-compose.yml) so that it has your End Model's environment variable.

Once you have an End Model that is performant, and you have specified the path to it, you can reload the deployment API by running these commands:
```shell
docker-compose stop
docker-compose --profile dev up -d
```
and visit http://localhost:80/docs to see the deployment API. You can use this API to get predictions on unseen datapoints, and then determine a method to load the JSON response into your data table in CrateDB. You now have predicted labels for your data and can perform any downstream analyses you need!

## Additional Installation Notes: ##

If you want to use CrateDB on your host machine but are experiencing issues, please go through these bootstrap checks,
as the host system must be configured correctly to use CrateDB with Docker.
https://crate.io/docs/crate/howtos/en/latest/admin/bootstrap-checks.html

Check this out if you are hosting CrateDB or another SQLAlchemy-based database on a remote server:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html

If you want to train the Label Model using CUDA tensors, then please refer to these resources:
https://developer.nvidia.com/cuda-toolkit
https://pytorch.org/docs/stable/cuda.html