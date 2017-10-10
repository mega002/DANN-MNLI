"""
TODO: update file description
Training script to train a model on a single genre from MultiNLI or on SNLI data.
The logs created during this training scheme have genre-specific statistics.
"""

######################### IMPORTS #############################

import tensorflow as tf
import os
import importlib
import random
from util import logger
from models.bilstm5_nl import *
import util.parameters5 as params
from util.data_processing5 import *
from util.evaluate import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")

training_labeled = []
training_unlabeled = []
for genre in GENRE_MAP:
    if GENRE_MAP[genre] < 5:
        training_data = load_nli_data_genre(FIXED_PARAMETERS["training_mnli"], genre, snli=False)
        split_idx = len(training_data) / 2
        training_labeled.extend(training_data[:split_idx])
        training_unlabeled.extend(training_data[split_idx:])
        logger.Log("Added genre %s to training set: %d labeled samples, %d unlabeled samples" % (genre, split_idx, len(training_data)-split_idx))

random.seed(5)
random.shuffle(training_labeled)
random.shuffle(training_unlabeled)

dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

logger.Log("Loading embeddings")
indices_to_words, word_indices = sentences_to_padded_index_sequences([training_labeled, training_unlabeled, dev_snli, dev_matched, dev_mismatched, test_snli])

loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


######################### MODEL #############################

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"]
        self.lmbda_init = FIXED_PARAMETERS["lmbda_init"]
        self.lmbda_rate = FIXED_PARAMETERS["lmbda_rate"]
        self.lmbda = self.lmbda_init

        logger.Log("Building model")

        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)
        self.total_cost = tf.add(self.model.pred_cost, tf.multiply(self.lmbda, self.model.domain_cost))

        tf.summary.scalar('total_loss', self.total_cost)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()


    def get_minibatch_samples(self, dataset, indices):
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        domains = [GENRE_MAP[dataset[i]['genre']] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres, domains


    def get_minibatch(self, labeled_samples, unlabeled_samples, idx):
        indices = range(self.batch_size * idx, self.batch_size * (idx + 1))
        s_premise_vectors, s_hypothesis_vectors, s_labels, s_genres, s_domains = self.get_minibatch_samples(labeled_samples, indices)
        t_premise_vectors, t_hypothesis_vectors, t_labels, t_genres, t_domains = self.get_minibatch_samples(unlabeled_samples, indices)
        premise_vectors = np.vstack((s_premise_vectors, t_premise_vectors))
        hypothesis_vectors = np.vstack((s_hypothesis_vectors, t_hypothesis_vectors))
        labels = s_labels + t_labels
        genres = s_genres + t_genres
        domains = s_domains + t_domains
        return premise_vectors, hypothesis_vectors, labels, genres, domains


    def train(self, training_labeled, training_unlabeled, dev_mat, dev_mismat, dev_snli):
        self.sess = tf.Session()
        train_writer = tf.summary.FileWriter('../logs')
        train_writer.add_graph(self.sess.graph)
        summary_op = tf.summary.merge_all()
        self.sess.run(self.init)

        self.step = 1
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_train_labeled_acc = 0.
        self.last_train_labeled_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore best-checkpoint if it exists.
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        # TODO: restore lambda
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat = evaluate_classifier_genre(self.classify, dev_mismat, self.batch_size)
                best_dev_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_train_labeled_acc, train_labeled_cost = evaluate_classifier(self.classify, training_labeled[0:5000], self.batch_size)
                logger.Log("Restored best matched-dev acc: %r\n Restored best mismatched-dev acc: %r\n Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f" % (
                        self.best_dev_mat, best_dev_mismat, best_dev_snli, self.best_train_labeled_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)


        ### Training cycle
        logger.Log("Training...")

        while True:
            random.shuffle(training_labeled)
            random.shuffle(training_unlabeled)
            avg_cost = 0.
            labeled_batch = int(len(training_labeled) / self.batch_size)
            unlabeled_batch = int(len(training_unlabeled) / self.batch_size)
            total_batch = min(labeled_batch, unlabeled_batch)

            # Boolean stating that training has not been completed, 
            self.completed = False 
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, minibatch_domains = self.get_minibatch(
                    training_labeled, training_unlabeled, i)
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels,
                                self.model.d: minibatch_domains,
                                self.model.keep_rate_ph: self.keep_rate,
                                self.model.idx: self.batch_size,
                                self.model.train: True}
                _, total_cost, summary_results = self.sess.run([self.optimizer, self.total_cost, summary_op], feed_dict)
                train_writer.add_summary(summary_results, self.step)
                train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % self.step)

                # Since a single epoch can take a  ages for larger models (ESIM),
                #  we'll print accuracy every 50 steps
                if self.step % self.display_step_freq == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier_genre(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier_genre(self.classify, dev_mismat, self.batch_size)
                    dev_acc_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    train_labeled_acc, train_labeled_cost = evaluate_classifier(self.classify, training_labeled[0:5000], self.batch_size)
                    logger.Log("step: %d\tmatched-dev acc: %r\n\t\tmismatched-dev acc: %r\n\t\tsnli-dev acc: %f\tlabeled train acc: %f" % (
                            self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, train_labeled_acc))
                    logger.Log("step: %d\tmatched-dev cost: %f\tmismatched-dev cost: %f\tsnli-dev cost: %f\tlabeled train cost: %f" % (
                            self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, train_labeled_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    dev_acc_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_train_labeled_acc = train_labeled_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best dev accuracy: %f" % (self.best_dev_mat))

                self.step += 1

                # Compute average loss
                avg_cost += total_cost / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f\t Lambda: %f" %(self.epoch+1, avg_cost, self.lmbda))
            
            self.epoch += 1
            # simple lambda adaptation
            self.lmbda += self.lmbda_rate
            # TODO: complex lambda adaptation
            # Adaptation param and learning rate schedule as described in the paper
            #p = float(i) / num_steps, l = 2. / (1. + np.exp(-10. * p)) - 1
            self.last_train_labeled_acc[(self.epoch % 5) - 1] = train_labeled_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_labeled_acc) / (5 * min(self.last_train_labeled_acc)) - 1)

            if (progress < 0.1) or (self.step > self.best_step + 10000):
                logger.Log("Best match-dev accuracy: %s" % (self.best_dev_mat))
                logger.Log("MultiNLI train labeled accuracy: %s" % (self.best_train_labeled_acc))
                self.completed = True
                break

        train_writer.close()

    def classify(self, examples):
        # This classifies a list of examples
        if (test == True) or (self.completed == True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        genres = []
        for i in range(total_batch):
            indices = range(self.batch_size * i, self.batch_size*(i+1))
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, minibatch_domains = self.get_minibatch_samples(
                examples, indices)
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels,
                                self.model.d: minibatch_domains,
                                self.model.keep_rate_ph: 1.0,
                                self.model.idx: 0,
                                self.model.train: False}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.pred_logits, self.model.pred_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), cost


######################### EXECUTION #############################

classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

# While test-set isn't released, use dev-sets for testing
test_matched = dev_matched
test_mismatched = dev_mismatched


if test == False:
    classifier.train(training_labeled, training_unlabeled, dev_matched, dev_mismatched, dev_snli)
    logger.Log("Test acc on matched multiNLI: %s" %(evaluate_classifier(classifier.classify, \
    test_matched, FIXED_PARAMETERS["batch_size"]))[0])

    logger.Log("Test acc on mismatched multiNLI: %s" %(evaluate_classifier(classifier.classify, \
    test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])

    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify, \
        test_snli, FIXED_PARAMETERS["batch_size"]))[0])
else:
    logger.Log("Test acc on matched multiNLI: %s" %(evaluate_classifier(classifier.classify, \
    test_matched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on mismatched multiNLI: %s" %(evaluate_classifier(classifier.classify, \
        test_mismatched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify, \
        test_snli, FIXED_PARAMETERS["batch_size"])[0]))
    
    # Results by genre,
    logger.Log("Test acc on matched genres: %s" %(evaluate_classifier_genre(classifier.classify, \
        test_matched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on mismatched genres: %s" %(evaluate_classifier_genre(classifier.classify, \
        test_mismatched, FIXED_PARAMETERS["batch_size"])[0]))
  

  
  