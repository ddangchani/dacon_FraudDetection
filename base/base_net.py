from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import tensorflow as tf
from keras import Model, layers # tensorflow base
import keras

class BaseNet(Model):
    """
    Base network for DeepSAD, using simple neural network
    """
    def __init__(self, rep_dim = 10):
        """
        rep_dim : representation dimensionality 
                i.e. dim of the code layer or last layer
        """
        super().__init__() # keras.Model (부모클래스)

        self.rep_dim = rep_dim
        
        self.snn = keras.Sequential([
            layers.Input(shape = (1,)),
            layers.Dense(30, activation='selu'),
            layers.Dense(rep_dim, activation='selu')
        ])  # simple neural network

    def call(self, x):
        x = self.snn(x)
        return x
    
class BaseNet_decoder(Model):
    def __init__(self, rep_dim = 10):
        super().__init__()

        self.rep_dim = rep_dim
        self.desnn = keras.Sequential([
            layers.Dense(30, activation='selu'),
            layers.Dense(30, activation='sigmoid')
        ]) # decoder
    
    def call(self, x):
        x = self.desnn(x)
        return x