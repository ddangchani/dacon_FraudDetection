import tensorflow as tf

class DeepSAD(object):
    """ class for DeepSAD
    Attributes:
        eta : control the effect of labeled set
        c : hyperparameter center
        net_name : name of neural net
        net : neural network (phi)
        trainer : DeepSADTrainer
        
    """

    def __init__(self, eta : float = 1.0):
        
        self.eta = eta
        self.c = None
        self.net_name = None
        self.net = None
        self.trainer = None
        self.optimizer_name = None
        self.ae_net = None
        self.ae_trainer = None
        self.ae_optimizer_name = None

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
    
    def set_network(self, net_name):
        