import tensorflow as tf
import matplotlib as plt
import os
import tensorflow_similarity as tfsim

def train(model, train_dataset, validation_dataset, epochs, distance = "cosine", callbacks_flag = True):
    os.makedirs("./output")
    path_log = "./output/train.csv"
    filepath="./output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    csv_logger = tf.keras.callbacks.CSVLogger(path_log, separator=',', append=True)
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks_list = [checkpoint, csv_logger]


    distance = "cosine"  # @param ["cosine", "L2", "L1"]{allow-input: false} 
    loss = tfsim.losses.MultiSimilarityLoss(distance=distance)

    if callbacks_flag:
        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks = callbacks_list )
    else:
        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

    print(history.history)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.title(f"Loss: {loss.name} - LR: {LR}")
    plt.show()