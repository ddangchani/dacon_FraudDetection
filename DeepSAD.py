import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from base.base_net import BaseNet, BaseNet_decoder

class DeepSAD():
    """ class for DeepSAD
    Attributes:
        eta : control the effect of labeled set
        c : hyperparameter center
        model_name : name of neural net
        model : neural network (phi)
        trainer : DeepSADTrainer
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
        