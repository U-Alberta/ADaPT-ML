"""
https://keras.io/examples/nlp/bidirectional_lstm_imdb/
https://keras.io/models/model/
https://keras.io/models/model/


This model expects three int32 Tensors as input: numeric token ids, an input mask to hold out padding tokens, and input
types to mark different segments within one input (if any). The separate preprocessing model at
https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1 transforms plain text inputs into this format.
"""
import tensorflow as tf

INIT_PARAMS = {
    'units': 64,
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'recurrent_initializer': 'orthogonal',
    'bias_initializer': 'zeros',
    'unit_forget_bias': True,
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.5,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
}

COMPILE_PARAMS = {
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': 'accuracy',
    'loss_weights': None,
    'sample_weight_mode': None,
    'weighted_metrics': None,
    'target_tensors': None,
}

TRAIN_PARAMS = {
    'batch_size': 32,
    'epochs': 10,
    'verbose': 2,
    'callbacks': None,
    'validation_split': 0.0,
    'shuffle': True,
    'sample_weight': None,
    'initial_epoch': 0,
    'steps_per_epoch': None,
    'validation_steps': None,
    'validation_freq': 2,
    'max_queue_size': 10,
    'workers': 1,
    'use_multiprocessing': False,
}


def train_biLSTM(x_train, y_train, x_valid, y_valid):
    """
    https://keras.io/examples/imdb_bidirectional_lstm/
    :param x_train:
    :param probs:
    :return:
    """

    # get the class weight
    weight = {0: 0.1, 1: 0.45, 2: 0.45}

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(len(tokenizer.word_index)+1, embedding_matrix.shape[1],
    #                                     trainable=True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(**INIT_PARAMS)))
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(**COMPILE_PARAMS)
    model.summary()
    print("x_train shape: ", x_train.shape)
    onehot_y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)

    print("x_valid shape: ", x_valid.shape)
    onehot_y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=3)
    valid = (x_valid, onehot_y_valid)

    print('Training biLSTM ...')
    model.fit(x_train,
              onehot_y_train,
              validation_data=valid,
              class_weight=weight,
              **TRAIN_PARAMS)

    return model
