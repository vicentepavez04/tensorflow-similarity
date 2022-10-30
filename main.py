
from ctypes import util
import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt','--path_train', help='path to train dataset', required=True)
    parser.add_argument('-pv','--path_valid', help='path to validation dataset', required=True)
    parser.add_argument('-e','--epochs', help='number of epochs for training')
    

    args = vars(parser.parse_args())

    train_dataset, validation_dataset = utils.load_dataset.load(args.path_train, args.path_valid, batch_size=64)
    model = utils.load_model.load()
    utils.trainer.train(model, train_dataset, validation_dataset, args.epochs)
