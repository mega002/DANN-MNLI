import tensorflow as tf
from util import blocks
from util.flip_gradient import flip_gradient

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='premise_x')
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='hypothesis_x')
        self.y = tf.placeholder(tf.int32, [None], name='y')
        self.d = tf.placeholder(tf.int32, [None], name='d')
        self.keep_rate_ph = tf.placeholder(tf.float32, [], name='keep_rate_ph')
        self.idx = tf.placeholder(tf.int32, [], name='idx')
        self.train = tf.placeholder(tf.bool, [], name='train')

        ### Feature extractor
        with tf.variable_scope('feature_extractor'):
            self.E = tf.Variable(embeddings, trainable=emb_train, name='E')

            ## Fucntion for embedding lookup and dropout at embedding layer
            def emb_drop(x):
                emb = tf.nn.embedding_lookup(self.E, x)
                emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
                return emb_drop

            # Get lengths of unpadded sentences
            prem_seq_lengths, prem_mask = blocks.length(self.premise_x)
            hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)

            ### BiLSTM layer ###
            premise_in = emb_drop(self.premise_x)
            hypothesis_in = emb_drop(self.hypothesis_x)

            premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2, name='premise_bi')
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2, name='hypothesis_bi')

            #premise_final = blocks.last_output(premise_bi, prem_seq_lengths)
            #hypothesis_final =  blocks.last_output(hypothesis_bi, hyp_seq_lengths)

            ### Mean pooling
            premise_sum = tf.reduce_sum(premise_bi, 1)
            premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1), name='premise_ave')

            hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
            hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1), name='hypothesis_ave')

            ### Mou et al. concat layer ###
            diff = tf.subtract(premise_ave, hypothesis_ave, name='diff')
            mul = tf.multiply(premise_ave, hypothesis_ave, name='mul')
            self.h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1, name='h')

        ### Label predictor
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples differently, depending on train or test mode.
            all_samples = lambda: self.h
            source_samples = lambda: tf.slice(self.h, [0, 0], [self.idx, -1])
            classify_samples = tf.cond(self.train, source_samples, all_samples)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0], [self.idx])
            classify_labels = tf.cond(self.train, source_labels, all_labels)

            # Variables
            self.W_pred_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1), name='W_pred_mlp')
            self.b_pred_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name='b_pred_mlp')
            self.W_pred_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1), name='W_pred_cl')
            self.b_pred_cl = tf.Variable(tf.random_normal([3], stddev=0.1), name='b_pred_cl')

            # MLP layer
            h_pred_mlp = tf.nn.relu(tf.matmul(classify_samples, self.W_pred_mlp) + self.b_pred_mlp)

            # Dropout applied to classifier
            h_pred_drop = tf.nn.dropout(h_pred_mlp, self.keep_rate_ph)

            # Get prediction
            self.pred_logits = tf.matmul(h_pred_drop, self.W_pred_cl) + self.b_pred_cl

            # Define the cost function
            self.pred_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classify_labels, logits=self.pred_logits))
            tf.summary.scalar('pred_loss', self.pred_cost)

        ### Domain classifier
        with tf.variable_scope('domain_classifier'):
            # Flip the gradient when backpropagating through this operation
            h_ = flip_gradient(self.h)

            # Variables
            self.W_domain_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1), name='W_domain_mlp')
            self.b_domain_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name='b_domain_mlp')
            self.W_domain_cl = tf.Variable(tf.random_normal([self.dim, 6], stddev=0.1), name='W_domain_cl')
            self.b_domain_cl = tf.Variable(tf.random_normal([6], stddev=0.1), name='b_domain_cl')

            # MLP layer
            h_domain_mlp = tf.nn.relu(tf.add(tf.matmul(h_, self.W_domain_mlp), self.b_domain_mlp))

            # Dropout applied to classifier
            h_domain_drop = tf.nn.dropout(h_domain_mlp, self.keep_rate_ph)

            # Get prediction
            self.domain_logits = tf.add(tf.matmul(h_domain_drop, self.W_domain_cl), self.b_domain_cl)
            domain_probs = tf.nn.softmax(self.domain_logits)
            #self.domain_neg_entropy = tf.reduce_mean(tf.reduce_sum(tf.multiply(domain_probs, log2(domain_probs)), axis=1, name='domain_neg_ent'))
            self.domain_neg_entropy = tf.reduce_mean(tf.reduce_sum(
                                        tf.multiply(domain_probs, tf.log(tf.clip_by_value(domain_probs, 1e-10, 1.0))),
                                        axis=1, name='domain_neg_ent'))
            tf.summary.scalar('domain_neg_entropy', self.domain_neg_entropy)

            # Define the cost function
            self.domain_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.d, logits=self.domain_logits))
            tf.summary.scalar('domain_loss', self.domain_cost)

'''
def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
'''