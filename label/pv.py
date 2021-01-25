import logging
import sys
import mlflow
from mlflow.models.signature import infer_signature
import json
from glob import glob
import requests

from label.lfs.pv import pv_lfs
from label.lfs.pv import ValueLabel

from label import parsed_args, model, evaluate, tracking

REGISTERED_MODEL_NAME = 'PersonalValuesLabelModel'
LF_COLUMNS = ['id', 'tweet_pv_words_count']


def main():

    with mlflow.start_run():

        # get the needed information for the pv lfs
        logging.info("Getting information for lfs ...")
        train_df = model.load_lf_info(LF_COLUMNS)

        # create the label matrix
        logging.info("Creating label matrix ...")
        train_L = model.create_label_matrix(train_df, pv_lfs)

        # train the label model
        logging.info("Training label model ...")
        label_model = model.train_label_model(train_L, ValueLabel)

        # evaluate the label model
        logging.info("Predicting multilabels ...")
        labeled_train_df = model.apply_label_preds(train_L, label_model, ValueLabel, parsed_args.task)

        # validate the training data
        # print("Validating training data ...")
        # model.validate_training_data(labeled_train_df, ValueLabel)

        if parsed_args.dev_data:
            # TODO: figure out how do get the label studio project uuid and switch then export the completions
            requests.get('http://129.128.215.241:8080/api/project-switch/?project_uuid=32d6ccea-2b41-479c-8272-a4ebfdd1d5aa')
            metrics = evaluate.multilabel_summary(labeled_train_df, label_model)
            pass

        logging.info("Saving ...")
        evaluate.lf_summary(train_L, pv_lfs, label_model)

        input_example = train_L[:5, :]

        tracking.log(model.train_params_dict(label_model),
                     metrics,
                     input_example,
                     REGISTERED_MODEL_NAME,
                     label_model)


if __name__ == '__main__':
    main()
