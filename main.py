
import utils.load_dataset as load_dataset
import utils.load_model as load_model
import utils.trainer as trainer 
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt','--path_train', type=str, help='path to train dataset', required=True)
    parser.add_argument('-pv','--path_valid', type=str, help='path to validation dataset', required=True)
    parser.add_argument('-e','--epochs', type=int, help='number of epochs for training')
    args = parser.parse_args()

    train_dataset, validation_dataset = load_dataset.load(args.path_train, args.path_valid, batch_size=64)
    model = load_model.load()
    trainer.train(model, train_dataset, validation_dataset, args.epochs)
