# ADaPT-ML

A Data Programming Template for Machine Learning

## Installation and Usage Guidelines ##

Follow these guidelines to set up ADaPT-ML on your machine and to see how you can add new classification tasks to the system. 

Before getting started, please make sure you are familiar with the following:
- [Docker](https://docs.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
  - working with environment variables, volumes, Dockerfiles, and the main docker-compose commands
- [Label Studio v1.0](https://labelstud.io/)
  - Labeling Config files
- [Snorkel v0.9.7](https://www.snorkel.org/get-started/)
  - writing Labeling Functions, Label Matrix, Label Model
- [MLflow v1.19.0](https://mlflow.org/)
  - MLflow Projects, MLflow Tracking, MLflow Models, Model Registry
- [FastAPI v0.68.1](https://fastapi.tiangolo.com/)
  - endpoints, Pydantic, requests, JSON

Now that you are familiar with the concepts, terminology, and tools that make up ADaPT-ML, let's get started!

### Installing Docker and Docker Compose ###
Docker and Docker Compose are required to use ADaPT-ML. Please follow the links for each and review the installation instructions, and make sure that they are installed on the host machine where you will be cloning this repository to.

### Setting up the environment variables for Docker Compose ###
Create a `.env` file in the repository's root directory with the environment variables listed below or add them to the host machine's environment variable list, and define these variables based off of the description listed for each one:
```shell
# The root directory for storing all Label Studio projects (including tasks and exported annotations) and all data programming / modelling artifacts and experiment runs (was set to ./example_data for the example use case).
DATA_PATH=

# The root directory for CrateDB (was set to ./crate for the example use case). If your SQLAlchemy database is stored on a remote server, then you can ignore this and delete the cratedb service from docker-compose.yml. If you want your database to be set up remotely but have not created it yet, then you can extract the cratedb service from docker-compose.yml and use it to set CrateDB up on the remote server and set this variable.
DB_DATA_PATH=

# Feel free to rename the Label Studio directory, but remember to change it in docker-compose.yml if you do.
LS_PATH=${DATA_PATH}/ls

# Feel free to rename the Label Studio data directory, but remember to change it in docker-compose.yml if you do.
LS_DATA_PATH=${LS_PATH}/data

# Feel free to rename the Label Studio tasks directory, but remember to change it in docker-compose.yml if you do.
LS_TASKS_PATH=${LS_PATH}/tasks

# Feel free to rename the Label Studio annotations directory, but remember to change it in docker-compose.yml if you do.
LS_ANNOTATIONS_PATH=${LS_PATH}/annotations

# Feel free to rename the data programming directory, but remember to change it in docker-compose.yml if you do.
DP_DATA_PATH=${DATA_PATH}/dp

# Feel free to rename the modelling directory, but remember to change it in docker-compose.yml if you do.
MODELLING_DATA_PATH=${DATA_PATH}/m

# If your CrateDB database is stored on a remote server, then change crate-db to the IP address of the host machine. If your database is supported by SQLAlchemy but is not specifically CrateDB, then refer to [2] at the bottom of this README.
DATABASE_IP=crate://crate-db:4200

# Feel free to rename the data programming MLflow database, or leave it as-is.
DP_MYSQL_DATABASE=mlruns.db

# Set the following username, password, and root password variables to your preferences.
DP_MYSQL_USER=
DP_MYSQL_PASSWORD=
DP_MYSQL_ROOT_PASSWORD=

# Leave these variables as-is.
DP_HOST_NAME=dp-mlflow-db
DP_MLFLOW_TRACKING_URI=mysql+pymysql://${DP_MYSQL_USER}:${DP_MYSQL_PASSWORD}@${DP_HOST_NAME}:3306/${DP_MYSQL_DATABASE}

# Feel free to rename the modelling MLflow database, or leave it as-is.
MODELLING_MYSQL_DATABASE=mlruns.db

# Set the following username, password, and root password variables to your preferences.
MODELLING_MYSQL_USER=
MODELLING_MYSQL_PASSWORD=
MODELLING_MYSQL_ROOT_PASSWORD=

# Leave these variables as-is.
MODELLING_HOST_NAME=modelling-mlflow-db
MODELLING_MLFLOW_TRACKING_URI=mysql+pymysql://${MODELLING_MYSQL_USER}:${MODELLING_MYSQL_PASSWORD}@${MODELLING_HOST_NAME}:3306/${MODELLING_MYSQL_DATABASE}

# If you want to run through the example use case, then leave these variables as-is and make sure that DATA_PATH and DB_DATA_PATH are set to ./example_data and ./crate respectively. Otherwise, these variables are not required to be set. Once you have a model of your own ready to be deployed, set these according to the path to the python_model.pkl artifact displayed in MLflow (see [3])
MULTICLASS_EXAMPLE_MODEL_PATH=/mlruns/multiclass_model.pkl
MULTILABEL_EXAMPLE_MODEL_PATH=/mlruns/multilabel_model.pkl
```

### Adding a new classification task ###

Setting up ADaPT-ML to work with a new classification task requires adding new Python modules and editing some existing files. You can follow along with the Example Use Case to get an overview, but the details are as follows, file-by-file:

#### [Define your class names](data-programming/label/lfs/__init__.py) ####

This is where you need to define the Class that will hold both the name of each class and the number representing that class, which the Labeling Functions will use to vote, and which will ultimately make up the Label Matrix.

#### [Write your Labeling Functions](data-programming/label/lfs/your_new_task.py) ####

This module that you can name after your new classification task is where you will write your Labeling Functions, and create a function called `get_lfs` that will produce an iterable containing all of the Labeling Functions you have defined.

#### [Create your main function as an MLflow endpoint](data-programming/label/your_new_task.py) ####

This is the main module for your new task. You will need to import the Class you defined in [this step](#define-your-class-namesdata-programminglabellfs__init__py) and the `get_lfs` function defined in [this step](#write-your-labeling-functionsdata-programminglabellfsyour_new_taskpy). You will also need to create a name for the Label Model that will be specific to your new task, and a dictionary with the names of the columns holding the features extracted for use with the Labeling Functions you defined as keys and any functions necessary to properly transform or unpack the featurized data point as values.

## Example Usage ##
Our example use case is to develop a model that can predict whether a data point is about a cat, dog, bird, horse, or snake. This task has been divided into a multiclass setting, where there is only one possible class that the data point can belong to, and a multilabel setting, where one data point can belong to one or many classes, to demonstrate how to handle both tasks. We do not have an existing annotated dataset for this classification task, so the first step will be to create one. When you first get started, you will need to gather the appropriate data for your task, and featurize it in two ways:
1. Decide on which features you would pull out to assist in annotation if you were going to manually assign classes to the datapoints.
2. Decide on how you would represent the datapoints as feature vectors for the end model.

In this use case, I manually created data points with only a text component to keep it simple, but consider the tweets 1a-1e in the diagram below. Many of them have both text and images that can provide information for more accurate classification. Let's run through each datapoint:
1a has a textual component with the keyword "Birds", and an image component that is a painting where birds can be identified.
1b has a textual component with the keyword "horse", and an image component that does not on its own provide information that it is related to horses.
1c only has a textual component with the keyword "dogs".
1e has a textual component that has the keyword "snake", but it is actually not about the animal. The image component does not on its own provide information that is related to snakes.

![Step One](./graphics/step_1.jpg)

## Community Guidelines ##

Follow these guidelines to see where you can contribute new modules to expand the system's functionality and adaptability. The following items are on ADaPT-ML's "wish list":




## Additional Installation Notes: ##
If you want to use CrateDB on your host machine but are experiencing issues, please go through these bootstrap checks,
as the host system must be configured correctly to use CrateDB with Docker.

https://crate.io/docs/crate/howtos/en/latest/admin/bootstrap-checks.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html