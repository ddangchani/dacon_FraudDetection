from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import tensorflow as tf
from keras import Model, layers # tensorflow base
import keras

class BaseNet(Model):
    """
    Base network for DeepSAD, using simple neural network
    """
    def __init__(self, rep_dim = 128):
        """
        rep_dim : representation dimensionality 
                i.e. dim of the code layer or last layer
        """
        super().__init__() # keras.Model (부모클래스)

        self.rep_dim = rep_dim
        
        self.snn = keras.Sequential([
            layers.Input(shape = (1,)),
            layers.Dense(64, activation='linear'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(128, activation='linear'),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])  # simple neural network

    def call(self, x):
        x = self.snn(x)
        return x
    
class BaseNet_decoder(Model):
    def __init__(self, rep_dim = 128):
        super().__init__()

        self.rep_dim = rep_dim
        self.desnn = keras.Sequential([
            layers.Dense(64, activation='linear'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(30, activation='linear')
        ]) # decoder
    
    def call(self, x):
        x = self.desnn(x)
        return x