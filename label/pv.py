import logging
import os
import json
import mlflow

from label import TRAIN_DF, parsed_args, model, evaluate, tracking
from label.lfs.pv import ValueLabel
from label.lfs.pv import pv_lfs

REGISTERED_MODEL_NAME = 'PersonalValuesLabelModel'
LF_FEATURES = {'text_pv_freq': json.loads}
PV_DEV_COMPLETIONS_DIRECTORY = os.path.join('/labeled_data', 'pv_completions', '*')


def main():
    with mlflow.start_run():
        # get the needed information for the pv lfs
        logging.info("Getting information for lfs ...")
        train_df = model.load_lf_info(TRAIN_DF, LF_FEATURES)

        if parsed_args.dev_data:
            logging.info("Getting development data if available ...")
            dev_df = model.load_lf_info(evaluate.get_dev_df(PV_DEV_COMPLETIONS_DIRECTORY), LF_FEATURES)
            dev_true = dev_df.gold_label.tolist()
        else:
            logging.info("Skipping development data ...")
            dev_df = None
            dev_true = None

        # create the label matrix
        logging.info("Creating label matrix ...")
        train_L = model.create_label_matrix(train_df, pv_lfs)
        dev_L = model.create_label_matrix(dev_df, pv_lfs)

        # train the label model
        logging.info("Training label model ...")
        label_model = model.train_label_model(train_L, dev_true, ValueLabel)

        # evaluate the label model
        logging.info("Predicting {} ...".format(parsed_args.task))
        labeled_train_df = model.apply_label_preds(train_df, train_L, label_model, ValueLabel, parsed_args.task)
        labeled_dev_df = model.apply_label_preds(dev_df, dev_L, label_model, ValueLabel, parsed_args.task)

        train_pred = labeled_train_df.label.tolist()
        try:
            dev_pred = labeled_dev_df.label.tolist()
        except:
            dev_pred = None

        # validate the training data
        print("Validating training data ...")
        model.validate_training_data(labeled_train_df, ValueLabel)

        logging.info("Evaluating ...")
        evaluate.lf_summary(train_L, dev_L, pv_lfs, label_model, dev_true)
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
