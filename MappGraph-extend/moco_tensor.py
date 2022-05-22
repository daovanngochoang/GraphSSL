import numpy as np
import tensorflow as tf
from tensorflow import keras


class MocoTensor (tf.keras.Model):
    def __init__(self,  base_encoder, dim=128, K=512, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MocoTensor, self).__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        
        # initialize the encoders
        self.query_encoder = base_encoder
         # initialize the key encoder weights by copy the query encoder weights
        self.key_encoder = keras.models.clone_model(self.query_encoder)
        tf.stop_gradient(self.key_encoder.weights)
        
        # queue pointer and queue initialize with normalization L2
        self.queue = tf.Variable(tf.random.normal(shape=(dim, K), dtype=tf.float64), trainable=False)
        self.queue_pointer = tf.Variable(0, trainable=False)
        
        
     
        
        
        
    def momentum_update_key_encoder(self):
        
        # get the weights from encoders
        query_encoder_weights = self.query_encoder.get_weights()
        key_encoder_weights = self.key_encoder.get_weights()
        
        '''
            momentum update key encoder weights based on query encoder weights by the formula:
                key_encoder_weights = key_encoder_weights * m + query_encoder_weights * (1 - m)
        '''
        for i in range(len(key_encoder_weights)):
            key_encoder_weights[i] = key_encoder_weights[i] * self.m + query_encoder_weights[i] * (1. - self.m)
        
        # set the new weights to key encoder
        self.key_encoder.set_weights(key_encoder_weights)
        
        
        
        
    def dequeue_and_enqueue(self, keys):
        
        batch_size =  keys.shape[0] 

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = np.transpose(keys)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_pointer[0] = ptr
        
        
        
        
    def call(self, query_graphs, key_graphs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        encoded_queries, _ = self.encoder_q(query_graphs)  # queries: NxC
        encoded_queries = keras.utils.normalize(encoded_queries, dim=1)
        
        self.momentum_update_key_encoder()
        
        # compute key features
        encoded_keys, _ = self.encoder_k(key_graphs)  # keys: NxC
        encoded_keys = keras.utils.normalize(encoded_keys, dim=1)
        
        # a single query and key has dimesion of 1xC
        
        # loss
        l_pos = tf.reshape(tf.einsum('nc,nc->n', encoded_queries, encoded_keys), (-1, 1))  # nx1
        l_neg = tf.einsum('nc,kc->nk', encoded_queries, self.queue)  # nxK
         
        
        logits = tf.concat([l_pos, l_neg], axis=1) # nx(1+k)
        logits = logits * (1 / self.T)
        labels = tf.zeros(logits.shape[0], dtype=tf.int64)  #n

        
        # dequeue and enqueue
        self.dequeue_and_enqueue(encoded_keys)
        
        return logits, labels
        
        
        
        
            
        
        
        