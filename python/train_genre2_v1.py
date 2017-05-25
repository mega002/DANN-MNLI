"""
Training script to train a model on a single genre from MultiNLI or on SNLI data.
The logs created during this training scheme have genre-specific statistics.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
from models.bilstm2 import *
import util.parameters2 as params
from util.data_processing2 import *
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
sgenres = ['travel', 'fiction', 'slate', 'telephone', 'government', 'snli']
tgenres = ['travel', 'fiction', 'slate', 'telephone', 'government', 'snli',
           'facetoface', 'oup', 'letters', 'nineeleven', 'verbatim']

alpha = FIXED_PARAMETERS["alpha"]
sgenre = FIXED_PARAMETERS["source_genre"]
tgenre = FIXED_PARAMETERS["target_genre"]

# TODO: make script stop in parameter.py if genre name is invalid.
# TODO: take care of the case we train on a dev domain
if sgenre not in sgenres:
    logger.Log("Invalid source genre")
    exit()
if tgenre not in tgenres:
    logger.Log("Invalid target genre")
    exit()
logger.Log("Training on source genre %s and target genre %s" %(sgenre, tgenre))

if sgenre == "snli":
    training_sgenre = load_nli_data_genre(FIXED_PARAMETERS["training_snli"], sgenre, snli=True)
    beta = int(len(training_sgenre) * alpha)
    training_sgenre = random.sample(training_sgenre, beta)
else:
    training_sgenre = load_nli_data_genre(FIXED_PARAMETERS["training_mnli"], sgenre, snli=False)

if tgenre == "snli":
    training_tgenre = load_nli_data_genre(FIXED_PARAMETERS["training_snli"], tgenre, snli=True)
    beta = int(len(training_tgenre) * alpha)
    training_tgenre = random.sample(training_tgenre, beta)
else:
    training_tgenre = load_nli_data_genre(FIXED_PARAMETERS["training_mnli"], tgenre, snli=False)

dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

logger.Log("Loading embeddings")
indices_to_words, word_indices = sentences_to_padded_index_sequences([training_sgenre, training_tgenre, dev_snli, dev_matched, dev_mismatched, test_snli])

loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

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
        self.sgenre = FIXED_PARAMETERS["source_genre"]
        self.tgenre = FIXED_PARAMETERS["target_genre"]
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.Log("Building model")
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()


    def get_minibatch_samples(self, dataset, indices):
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        domains = [0 if dataset[i]['domain']==GENRE_MAP[sgenre] else 1 for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres, domains

    def get_minibatch(self, source_samples, target_samples, idx):
        indices = range(self.batch_size * idx, self.batch_size * (idx + 1))
        s_premise_vectors, s_hypothesis_vectors, s_labels, s_genres, s_domains = self.get_minibatch_samples(source_samples, indices)
        t_premise_vectors, t_hypothesis_vectors, t_labels, t_genres, t_domains = self.get_minibatch_samples(target_samples, indices)
        premise_vectors = np.vstack((s_premise_vectors, t_premise_vectors))
        hypothesis_vectors = np.vstack((s_hypothesis_vectors, t_hypothesis_vectors))
        labels = s_labels + t_labels
        genres = s_genres + t_genres
        domains = s_domains + t_domains
        return premise_vectors, hypothesis_vectors, labels, genres, domains


    def train(self, training_sgenre, training_tgenre, dev_mat, dev_mismat, dev_snli):
        self.sess = tf.Session()
        train_writer = tf.summary.FileWriter('../logs')
        train_writer.add_graph(self.sess.graph)
        summary_op = tf.summary.merge_all()
        self.sess.run(self.init)

        self.step = 1
        self.epoch = 0
        self.best_dev_sgenre_acc = 0.
        self.best_train_sgenre_acc = 0.
        self.last_train_sgenre_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore best-checkpoint if it exists.
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        # TODO: restore lambda
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                if sgenre == 'snli':
                    dev_acc, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    self.best_dev_sgenre_acc = dev_acc
                else:
                    best_dev_mat, dev_cost_mat = evaluate_classifier_genre(self.classify, dev_mat, self.batch_size)
                    self.best_dev_sgenre_acc = best_dev_mat[sgenre]
                self.best_train_sgenre_acc, mtrain_cost = evaluate_classifier(self.classify, training_sgenre[0:5000], self.batch_size)

                logger.Log("Restored best dev sgenre acc: %f\n Restored best train sgenre acc: %f" % (self.best_dev_sgenre_acc, self.best_train_sgenre_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)


        ### Training cycle
        logger.Log("Training...")

        while True:
            random.shuffle(training_sgenre)
            random.shuffle(training_tgenre)
            avg_cost = 0.
            #total_batch = int((len(training_sgenre)+len(training_tgenre)) / self.batch_size)
            source_batch = int(len(training_sgenre) / self.batch_size)
            target_batch = int(len(training_tgenre) / self.batch_size)
            total_batch = min(source_batch, target_batch)

            # Boolean stating that training has not been completed, 
            self.completed = False 
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, minibatch_domains = self.get_minibatch(
                    training_sgenre, training_tgenre, i)
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels,
                                self.model.d: minibatch_domains,
                                self.model.keep_rate_ph: self.keep_rate,
                                self.model.idx: self.batch_size,
                                self.model.train: True,
                                self.model.lmbda: self.lmbda}
                _, total_cost, summary_results = self.sess.run([self.optimizer, self.model.total_cost, summary_op], feed_dict)
                train_writer.add_summary(summary_results, self.step)
                train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % self.step)

                # Since a single epoch can take a  ages for larger models (ESIM),
                #  we'll print accuracy every 50 steps
                if self.step % self.display_step_freq == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier_genre(self.classify, dev_mat, self.batch_size)
                    if sgenre == 'snli':
                        dev_sgenre_acc, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    else:
                        dev_sgenre_acc = dev_acc_mat[sgenre]
                    if tgenre == 'snli':
                        dev_tgenre_acc, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    else:
                        dev_tgenre_acc = dev_acc_mat[tgenre]
                    #dev_acc_mismat, dev_cost_mismat = evaluate_classifier_genre(self.classify, dev_mismat, self.batch_size)
                    #dev_acc_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    train_sgenre_acc, train_sgenre_cost = evaluate_classifier(self.classify, training_sgenre[0:5000], self.batch_size)
                    train_tgenre_acc, train_tgenre_cost = evaluate_classifier(self.classify, training_tgenre[0:5000], self.batch_size)

                    logger.Log("Step: %i\t dev-sgenre acc: %f\t dev-tgenre acc: %f\t train-sgenre acc: %f\t train-tgenre acc: %f" %(self.step, dev_sgenre_acc, dev_tgenre_acc, train_sgenre_acc, train_tgenre_acc))
                    logger.Log("Step: %i\t dev-matched cost: %f\t train-sgenre cost: %f\t train-tgenre cost: %f" %(self.step, dev_cost_mat, train_sgenre_cost, train_tgenre_cost))
                    #logger.Log("Step: %i\t Dev-genre acc: %f\t Dev-mrest acc: %r\t Dev-mmrest acc: %r\t Dev-SNLI acc: %f\t Genre train acc: %f" %(self.step, dev_acc, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc))
                    #logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t Genre train cost: %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_sgenre_acc / dev_sgenre_acc)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_sgenre_acc = dev_sgenre_acc
                        self.best_train_sgenre_acc = train_sgenre_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best dev accuracy: %f" % (self.best_dev_sgenre_acc))

                self.step += 1

                # Compute average loss
                avg_cost += total_cost / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f\t Lambda: %f" %(self.epoch+1, avg_cost, self.lmbda))
            
            self.epoch += 1
            self.lmbda += self.lmbda_rate
            self.last_train_sgenre_acc[(self.epoch % 5) - 1] = train_sgenre_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_sgenre_acc) / (5 * min(self.last_train_sgenre_acc)) - 1)

            if (progress < 0.1) or (self.step > self.best_step + 10000):
                logger.Log("Best matched-dev accuracy: %s" % (self.best_dev_sgenre_acc))
                logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
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
                                self.model.train: False,
                                self.model.lmbda: self.lmbda_init}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.pred_logits, self.model.pred_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), cost


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
    classifier.train(training_sgenre, training_tgenre, dev_matched, dev_mismatched, dev_snli)
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
  

  
  