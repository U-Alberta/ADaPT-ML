import logging
import sys

import mlflow
from label import (TRAIN_DF, TRAIN_MATRIX_FILENAME, TRAINING_DATA_FILENAME,
                   TRAINING_DATA_HTML_FILENAME, DEV_DF_FILENAME, DEV_DF_HTML_FILENAME, DEV_MATRIX_FILENAME,
                   LABEL_MODEL_FILENAME, parsed_args, procedure, evaluate, tracking)


def start(registered_model_name, lf_features, dev_annotations_path, get_lfs, class_labels, parallel=False):
    with mlflow.start_run():
        run = mlflow.active_run()
        logging.info("Active run_id: {}".format(run.info.run_id))

        # get the needed information for the pv lfs
        logging.info("Getting information for lfs ...")
        train_df = procedure.load_lf_info(TRAIN_DF, lf_features)

        if parsed_args.dev_data:
            logging.info("Getting development data if available ...")
            dev_df = procedure.load_lf_info(evaluate.get_dev_df(dev_annotations_path), lf_features)
            dev_true = dev_df.gold_label.tolist()
        else:
            logging.info("Skipping development data ...")
            dev_df = None
            dev_true = None
        # create the label matrix
        lfs = get_lfs()
        logging.info("Creating label matrix ...")
        try:
            train_L = procedure.create_label_matrix(train_df, lfs, parallel)
            procedure.save_label_matrix(train_L, TRAIN_MATRIX_FILENAME)
        except Exception as e:
            msg = "Unable to create train label matrix:\n{}\nStopping.".format(e.args)
            logging.error(msg)
            sys.exit(msg)
        try:
            dev_L = procedure.create_label_matrix(dev_df, lfs, parallel)
            procedure.save_label_matrix(dev_L, DEV_MATRIX_FILENAME)
        except Exception as e:
            dev_L = None
            if dev_df:
                msg = "Unable to create dev label matrix:\n{}\nProceeding without class balance.".format(e.args)
                logging.warning(msg)

        # train the label model
        logging.info("Training label model ...")
        try:
            label_model = procedure.train_label_model(train_L, dev_true, class_labels)
            label_model.save(LABEL_MODEL_FILENAME)
        except Exception as e:
            msg = "Unable to train label model:\n{}\nStopping.".format(e.args)
            logging.error(msg)
            sys.exit(msg)

        # use the label model to label the data
        logging.info("Predicting {} ...".format(parsed_args.task))
        labeled_train_df = procedure.apply_label_preds(train_df, train_L, label_model, class_labels, parsed_args.task)
        procedure.save_df(labeled_train_df[['table', 'id', 'label', 'label_probs']],
                          TRAINING_DATA_FILENAME, TRAINING_DATA_HTML_FILENAME)
        try:
            labeled_dev_df = procedure.apply_label_preds(dev_df, dev_L, label_model, class_labels, parsed_args.task)
            procedure.save_df(labeled_dev_df[['table', 'id', 'gold_label', 'label', 'label_probs']],
                              DEV_DF_FILENAME, DEV_DF_HTML_FILENAME)
            dev_pred = labeled_dev_df.label.tolist()
            dev_true = labeled_dev_df.gold_label.tolist()
        except Exception as e:
            dev_pred = None
            dev_true = None
            if dev_df:
                msg = "Unable to create final dev df:\n{}\nProceeding without evaluation.".format(e.args)
                logging.warning(msg)

        # validate the training data
        logging.info("Validating training data ...")
        procedure.validate_training_data(labeled_train_df, class_labels)

        # evaluate the labeling functions and label model predictions
        logging.info("Evaluating ...")
        if parsed_args.task == 'multiclass':
            try:
                dev_true = [label for sublist in dev_true for label in sublist]
                dev_pred = [label for sublist in dev_pred for label in sublist]
                dev_true_lfs = [class_labels[label].value for label in dev_true]
            except Exception as e:
                dev_true_lfs = None
                if dev_true and dev_pred:
                    logging.warning("Problem flattening development labels and predictions:\n{}\n".format(e.args))
            metrics = evaluate.multiclass_summary(train_L, dev_L, lfs, dev_true, dev_true_lfs, dev_pred, label_model)
        elif parsed_args.task == 'multilabel':
            metrics = evaluate.multilabel_summary(train_L, dev_L, lfs, dev_true, dev_pred, label_model)

        logging.info("Logging artifacts and saving ...")
        input_example = train_L[:5, :]
        tracking.log(metrics,
                     input_example,
                     registered_model_name,
                     label_model)
