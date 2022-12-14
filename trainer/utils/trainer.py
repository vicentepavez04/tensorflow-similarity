import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow_similarity as tfsim

def train(model, train_dataset, validation_dataset, epochs, distance = "cosine", callbacks_flag = True):
    os.makedirs("./output", exist_ok=True)
    path_log = "./output/train.csv"
    filepath="./output/weights-improvement-{epoch:02d}-{val_loss:.2f}"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_freq = 'epoch')
    csv_logger = tf.keras.callbacks.CSVLogger(path_log, separator=',', append=True)
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks_list = [checkpoint, csv_logger]
    distance = "cosine"  # @param ["cosine", "L2", "L1"]{allow-input: false} 
    loss = tfsim.losses.MultiSimilarityLoss(distance=distance)
    LR = 0.0001
    print("Compiling the model")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss)

    print("training .............")
    
    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks = callbacks_list )
    model.save('./output/final_weight')

    print("Model trained succesfully")
    
    print(history.history)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.title(f"Loss: {loss.name} - LR: {LR}")
    plt.savefig('./output/history.png')
    #plt.show()