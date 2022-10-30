import tensorflow as tf

def load(path_train, path_valid, batch_size=64, Img_size=(256, 256)):

    train_dataset = tf.keras.utils.image_dataset_from_directory(path_train,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=Img_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(path_valid,
                                                                    shuffle=True,
                                                                    batch_size=batch_size,
                                                                    image_size=Img_size)

    return train_dataset, validation_dataset
