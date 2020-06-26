from label import parsed_args, TRAIN_CSV_FILENAME
from label import matrix
from label import model


if __name__ == '__main__':

    # Step 1: create the label matrix
    if parsed_args.step == 0 or parsed_args.step == 1:
        train_df = matrix.create_df(TRAIN_CSV_FILENAME)
        L_train = matrix.create_label_matrix(train_df)
        matrix.save_label_matrix(L_train)
    else:
        L_train = matrix.load_label_matrix()

    # Step 2: train the label model
    if parsed_args.step == 0 or parsed_args.step == 2:
        label_model = model.train_label_model(L_train)
        model.save_label_model(label_model)
    else:
        label_model = model.load_label_model()
