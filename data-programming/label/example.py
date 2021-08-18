import logging
import os
import sys

import mlflow
from label import TRAIN_DF, DEV_DF_FILENAME, DEV_DF_HTML_FILENAME, parsed_args, procedure, evaluate, tracking
from label.lfs import ExampleLabels
from label.lfs.example import get_lfs

REGISTERED_MODEL_NAME = 'ExampleLabelModel'
LF_FEATURES = {'txt_clean_lemma': None}
EXAMPLE_DEV_COMPLETIONS_DIRECTORY = os.path.join('/labeled_data', 'eg_completions', '*')


def main():
    with mlflow.start_run():
        # get the needed information for the pv lfs
        logging.info("Getting information for lfs ...")
        train_df = procedure.load_lf_info(TRAIN_DF, LF_FEATURES)

        if parsed_args.dev_data:
            logging.info("Getting development data if available ...")
            dev_df = procedure.load_lf_info(evaluate.get_dev_df(EXAMPLE_DEV_COMPLETIONS_DIRECTORY), LF_FEATURES)
            procedure.save_df(dev_df, DEV_DF_FILENAME, DEV_DF_HTML_FILENAME)
            dev_true = dev_df.gold_label.tolist()
        else:
            logging.info("Skipping development data ...")
            dev_df = None
            dev_true = None

        # create the label matrix
        lfs = get_lfs()
        logging.info("Creating label matrix ...")
        try:
            train_L = procedure.create_label_matrix(train_df, lfs)
        except Exception as e:
            msg = "Unable to create train label matrix:\n{}\nStopping.".format(e.args)
            logging.error(msg)
            sys.exit(msg)
        try:
            dev_L = procedure.create_label_matrix(dev_df, lfs)
        except Exception as e:
            dev_L = None
            msg = "Unable to create dev label matrix:\n{}\nProceeding without class balance.".format(e.args)
            logging.warning(msg)

        # train the label model
        logging.info("Training label model ...")
        try:
            label_model = procedure.train_label_model(train_L, dev_true, ExampleLabels)
        except Exception as e:
            msg = "Unable to train label model:\n{}\nStopping.".format(e.args)
            logging.error(msg)
            sys.exit(msg)

        # use the label model to label the data
        logging.info("Predicting {} ...".format(parsed_args.task))
        labeled_train_df = procedure.apply_label_preds(train_df, train_L, label_model, ExampleLabels, parsed_args.task)
        try:
            labeled_dev_df = procedure.apply_label_preds(dev_df, dev_L, label_model, ExampleLabels, parsed_args.task)
            dev_pred = labeled_dev_df.label.tolist()
        except:
            dev_pred = None

        # validate the training data
        print("Validating training data ...")
        procedure.validate_training_data(labeled_train_df, ExampleLabels)

        # evaluate the labeling functions and label model predictions
        logging.info("Evaluating ...")
        evaluate.lf_summary(train_L, dev_L, lfs, label_model, dev_true)
        if parsed_args.task == 'multiclass':
            metrics = evaluate.multiclass_summary(dev_L, dev_true, dev_pred, label_model)
        elif parsed_args.task == 'multilabel':
            metrics = evaluate.multilabel_summary(dev_true, dev_pred)
        logging.info("Logging artifacts and saving ...")
        input_example = train_L[:5, :]
        tracking.log(metrics,
                     input_example,
                     REGISTERED_MODEL_NAME,
                     label_model)


if __name__ == '__main__':
    main()
