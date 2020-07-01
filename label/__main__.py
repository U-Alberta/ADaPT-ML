from label import (parsed_args, matrix, model,
                   TRAIN_CSV_FILENAME, TRAIN_DF_FILENAME, TRAIN_MATRIX_FILENAME,
                   LABEL_MODEL_FILENAME, TRAINING_DATA_FILENAME)


if __name__ == '__main__':

    train_df = matrix.create_df(TRAIN_CSV_FILENAME, TRAIN_DF_FILENAME)

    # Step 1: create the label matrix
    if parsed_args.step == 0 or parsed_args.step == 1:
        L_train = matrix.create_label_matrix(train_df, TRAIN_MATRIX_FILENAME)
    else:
        L_train = matrix.load_label_matrix(TRAIN_MATRIX_FILENAME)

    # Step 2: train the label model
    if parsed_args.step == 0 or parsed_args.step == 2:
        label_model = model.train_label_model(L_train, LABEL_MODEL_FILENAME)
    else:
        label_model = model.load_label_model(LABEL_MODEL_FILENAME)

    # Step 3: apply the label model to the training data
    if parsed_args.step == 0 or parsed_args.step == 3:
        filtered_train_df = model.apply_label_model(L_train, label_model, train_df, TRAINING_DATA_FILENAME)
