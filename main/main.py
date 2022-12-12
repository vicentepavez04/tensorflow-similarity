from math import cos
from itertools import chain
from pathlib import Path
import logging
# logging.basicConfig(level=logging.ERROR)
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_similarity as tfsim
from tensorflow_similarity.samplers import MultiShotMemorySampler  # sample data
from keras.applications import VGG16
import os 

class SimilarityModel:
    """ Use of tensorflow similarity to make inferences using siamese networks
    """
    def __init__(self, weight_path):
        """__init__ [summary]

        Args:
            model (str, optional): [description]. Defaults to None.
            weights (str, optional): [description]. Defaults to None.
            db_path (str, optional): [description]. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)



        # MODEL
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        x = vgg_model.output
        x = tf.keras.layers.Flatten()(x) # Flatten dimensions to for use in FC layers
        outputs = tfsim.layers.MetricEmbedding(64)(x)
        inputs=vgg_model.input
        model = tfsim.models.SimilarityModel(inputs, outputs)


        self.model = model.load_weights(weight_path)
        self.index_folder ="./index"
        os.makedirs(self.index_folder, exist_ok=True)


        # self.model = self._get_model() if load_model is False else self.model_load(self.model_path)
        #if load_weights: self.weights_load(self.weights_path)
        #if load_index: self.index_load(self.index_folder)
        self.model.index_summary()


    def model_load(self,model_path):
        try:
            reloaded_model = tf.keras.models.load_model(
                            str(model_path),
                            custom_objects={"SimilarityModel": tfsim.models.SimilarityModel},
                )
            # reloaded_model.load_index(str(model_path))
        except:
            raise RuntimeError('Failed to load model')
        return reloaded_model

    def index_load(self,index_path):
        try:
            self.model.load_index(str(index_path))
        except:
            raise RuntimeError('Failed to load index')




    @staticmethod
    def _recompile_model(model, LR:float=0.01,loss=None, distance='cosine', **kwargs):
        """_recompile_model Used in case one or multiple parameters of the training want to be changed
        TODO: usar funciones como entrada y **kwargs para definir los argumentos de dichas funciones
        Args:
            model : The model to be recompiled
            loss (optional): Function of the type tfsim.losses.{loss} Defaults to 'PNLoss'.
            distance (str, optional): [description]. Defaults to 'cosine'.
            **kwargs: extra arguments for the compile function
        Returns:
            [type]: [description]
        """
        loss = tfsim.losses.PNLoss(distance=distance) if loss is None else loss(distance=distance)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss, **kwargs)
        return model


    @staticmethod
    def scale_input(img: np.ndarray, size: int = 224) -> np.ndarray:
        img = tf.cast(img, dtype='uint8')
        size = tf.cast((size,size), dtype='int32')
        img = tf.image.resize(img, size)
        return np.asarray(img)
    @staticmethod
    def scale_input_(img: np.ndarray, size: int = 224) -> np.ndarray:
        img = tf.cast(img, dtype='uint8')
        w_1 = tf.image.resize_with_pad(tf.zeros_like(img)+255, size, size)
        img = tf.image.resize_with_pad(
            img, size, size)+(tf.zeros_like(w_1)+255 - w_1)
        return np.asarray(img)



    def gen_index(self, db_path=None, save=True, show_summary=True, append=False):

        for ref_folder in db_path.iterdir():
            
            for img in ref_folder.iterdir():
                if img.is_dir():
                    raise RuntimeError('Only images should be in the sku directory')
                x = cv2.cvtColor(self.scale_input(
                    cv2.imread(str(img))), cv2.COLOR_BGR2RGB)
                if x is None:
                    raise ValueError(f'{img} should be an image or is corrupt')

                y = ref_folder
                # self.model.index_single(x, y,verbose=False)
                self.model.index_single(x, y,build=False,verbose=False)

        self.model._index.search._build()
        if save:
            self.model.save_index(str(self.index_folder))
            # self.model.save(str(self.model_path), save_index=True)
        if show_summary: self.model.index_summary()
    def calibration(self,path,save = True):
        imgs = list(chain.from_iterable([list(dir_class.iterdir()) for dir_class in Path(path).iterdir()]))

        x_train = np.asarray([self.scale_input(cv2.cvtColor(
            cv2.imread(str(img)), cv2.COLOR_BGR2RGB)) for img in imgs])
        y_train = np.asarray([int(img.parent.stem) for img in imgs])
        self.model.index_summary()
        calibration = self.model.calibrate(
            x_train,
            y_train,
            extra_metrics=["precision", "recall", "binary_accuracy"],
            verbose=1)
        self.model.index_summary()
        # if save:
        #     self.model.save_index(str(self.index_folder))
            # self.model.save(str(self.model_path), save_index=True)
    def matching(self,img,cutpoint = "optimal"):
        img = self.scale_input(cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB))
        img = np.asarray([img])
        print('asd')
        match = self.model.match(img, cutpoint=cutpoint)
        print(match)
        match = str(match[0]).zfill(6)
        return match
    def matching_batch(self,img_batch,cutpoint = "optimal"):
        imgs = [self.scale_input(cv2.cvtColor(
            cv2.imread(str(img)), cv2.COLOR_BGR2RGB)) for img in img_batch]
        imgs = np.asarray(imgs)
        match = self.model.match(imgs, cutpoint=cutpoint)
        match = [str(m).zfill(6) for m in match]
        return match
        

if __name__ == "__main__":


    SimilarityModel(weight_path= "")
    SimilarityModel.gen_index(db_path= "")
    SimilarityModel.calibration(path = "")