from tabulate import tabulate
import tensorflow as tf
from keras.applications import VGG16
import tensorflow_similarity as tfsim

def load():
    print("loading model VGG16")
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = vgg_model.output
    x = tf.keras.layers.Flatten()(x) # Flatten dimensions to for use in FC layers
    outputs = tfsim.layers.MetricEmbedding(64)(x)
    inputs=vgg_model.input

    model = tfsim.models.SimilarityModel(inputs, outputs)
    print("Model loaded correctly")


    return model