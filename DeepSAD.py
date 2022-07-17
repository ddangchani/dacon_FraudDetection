import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score, f1_score
from base.base_net import BaseNet, BaseNet_decoder

class DeepSAD():
    """ class for DeepSAD
    Attributes:
        eta : control the effect of labeled set
        c : hyperparameter center
        model_name : name of neural net
        model : neural network (phi)
    """

    def __init__(self, config):
        
        self.eta = config['eta']
        self.c = None
        self.eps = 1e-6

        self.model_name = config['model_name']
        self.model = None

        self.trainer = None
        self.optimizer = None

        self.ae = None
        self.ae_optimizer = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }
    
    def set_network(self):
        """
        Set Encoder and Decoder
        """
        self.model = BaseNet()
    
    def set_decoder(self):
        self.decoder = BaseNet_decoder()
    
    def init_c(self, eps = 0.1):
        """
        initialize hypersphere center
        """
        print('Initialize hypersphere center c')
        n = 0
        c = tf.zeros(self.model.rep_dim)
        for inputs, labels in self.train_dataset:
            outputs = self.model(inputs) # encoder pass
            c += tf.reduce_sum(outputs, axis=0)
            n += inputs.shape[0]

        c /= n # average of summation
        c = tf.where((c>=0)&(c<eps), eps, c)
        c = tf.where((c<0)&(c>-eps), -eps, c)

        self.c = c

    def train(self, train_dataset, lr, epochs, beta1=0.9, beta2=0.999):

        self.train_dataset = train_dataset
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.optimizer = keras.optimizers.adam_v2.Adam(
            learning_rate=self.lr, beta_1=self.beta1, 
            beta_2=self.beta2, epsilon=1e-08)

        self.init_c() # hypersphere center initialize
        for epoch in range(epochs):
            for inputs, labels in self.train_dataset:
                loss = self.train_step(inputs, labels)

            print(f"epoch: {epoch+1}/{epochs}, loss: {loss}")

        return
    
    def train_step(self, inputs, labels): # Gradient Descent
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            dist = tf.reduce_sum((outputs-self.c)**2, axis=1) # norm-squared distance
            losses = tf.where( # tf.where > choose x if condition is True
                labels == 0,
                x = dist, # for unlabeled, take dist(squared norm)
                y = self.eta * ((dist + self.eps) ** tf.cast(labels, dtype=tf.float32))
                # for labeled, take second term
            )
            loss = tf.reduce_min(losses) # minimization

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss
    
    def pretrain(self, train_dataset, lr, epochs, beta1=0.9, beta2=0.999):

        print('Start Pretraining')
        self.train_dataset = train_dataset
        self.optimizer = keras.optimizers.adam_v2.Adam(
            learning_rate=self.lr, beta_1=self.beta1, 
            beta_2=self.beta2, epsilon=1e-08)
        self.set_decoder()
        for epoch in range(epochs):
            for inputs, labels in self.train_dataset:
                loss = self.pretrain_step(inputs, labels)
            print(f"epoch: {epoch+1}/{epochs}, ae_loss: {loss}")
    
    def pretrain_step(self, inputs, semi_labels):

        mse = keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            embeds = self.model(inputs, training=True)
            outputs = self.decoder(embeds, training=True)
            loss = mse(inputs, outputs)

        gradients = tape.gradient(loss, self.model.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables + self.decoder.trainable_variables))

        return loss