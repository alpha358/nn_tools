import keras


def decaying_optimizer(EPOCHS, TRAIN_SET_SIZE, BATCH_SIZE):
    '''
        initial_learning_rate = 1e-03
        final_learning_rate = 1e-05
    '''
    # TRAIN_SET_SIZE = (31 * 0.9) * 1e3 # approx training set size

    epochs = EPOCHS
    # how many updates in epoch, how many baches in epoch
    epoch_size = TRAIN_SET_SIZE / BATCH_SIZE

    initial_learning_rate = 1e-03
    final_learning_rate = 1e-05

    total_updates = epoch_size * epochs

    decay = ((1.0 / final_learning_rate *
              initial_learning_rate - 1.0) / total_updates)

    optimizer = keras.optimizers.Adam(lr=initial_learning_rate, decay=decay)

    return optimizer
