"""
Training script to train a model on a single genre from MultiNLI or on SNLI data.
The logs created during this training scheme have genre-specific statistics.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
from models.bilstm2_light import *
import util.parameters2 as params
from util.data_processing2 import *
from util.evaluate import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
acclogpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".acc.log"
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
        self.starter_learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"]
        self.lmbda_rate = FIXED_PARAMETERS["lmbda_rate"]
        self.lmbda = 0.
        self.sgenre = FIXED_PARAMETERS["source_genre"]
        self.tgenre = FIXED_PARAMETERS["target_genre"]
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.num_epochs = FIXED_PARAMETERS["num_epochs"]
        self.init_step = FIXED_PARAMETERS["init_step"]

        logger.Log("Building model")
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)
        self.total_cost = tf.add(self.model.pred_cost, self.model.domain_cost)

        # Perform gradient descent with Adam
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100000, 0.96)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_cost,
                                                                             global_step=self.global_step)

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


    def train(self, training_sgenre, training_tgenre, dev_mat, dev_mismat, dev_snli, acclog):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.completed = False
        self.step = 1
        self.epoch = 0
        source_batch = int(len(training_sgenre) / self.batch_size)
        target_batch = int(len(training_tgenre) / self.batch_size)
        self.total_batch = min(source_batch, target_batch)

        self.best_dev_tgenre_acc = 0.
        self.best_train_sgenre_acc = 0.
        self.last_train_sgenre_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore best-checkpoint if it exists.
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                if tgenre == 'snli':
                    dev_acc, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    self.best_dev_tgenre_acc = dev_acc
                else:
                    best_dev_mat, dev_cost_mat = evaluate_classifier_genre(self.classify, dev_mat, self.batch_size)
                    self.best_dev_tgenre_acc = best_dev_mat[tgenre]
                self.best_train_sgenre_acc, mtrain_cost = evaluate_classifier(self.classify, training_sgenre[0:5000], self.batch_size)

                logger.Log("Restored best dev tgenre acc: %f\n Restored best train sgenre acc: %f" % (self.best_dev_tgenre_acc, self.best_train_sgenre_acc))
                self.sess.close()
                self.sess = tf.Session()

            self.saver.restore(self.sess, ckpt_file)
            self.step = self.init_step
            self.epoch = int(self.init_step / self.total_batch)
            logger.Log("Model restored from file: %s Step: %d Epoch: %d" % (ckpt_file, self.step, self.epoch+1))


        ### Training loop
        logger.Log("Training...")

        while True:
            random.shuffle(training_sgenre)
            random.shuffle(training_tgenre)
            avg_cost = 0.
            
            # Loop over all batches in epoch
            for i in range(self.total_batch):
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
                                self.model.l: self.lmbda}
                _, total_cost = self.sess.run([self.optimizer, self.total_cost], feed_dict)

                # Check performance
                if self.step % self.display_step_freq == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier_genre(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier_genre(
                        self.classify, dev_mismat, self.batch_size)
                    dev_acc_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    train_sgenre_acc, train_sgenre_cost = evaluate_classifier(self.classify, training_sgenre[0:5000],
                                                                              self.batch_size)
                    train_tgenre_acc, train_tgenre_cost = evaluate_classifier(self.classify, training_tgenre[0:5000],
                                                                              self.batch_size)
                    if tgenre == 'snli':
                        dev_tgenre_acc = dev_acc_snli
                    else:
                        dev_tgenre_acc = dev_acc_mat[tgenre]

                    acclog.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        self.step, train_sgenre_acc, train_tgenre_acc, dev_acc_mat["travel"],
                        dev_acc_mat["fiction"], dev_acc_mat["slate"],
                        dev_acc_mat["telephone"], dev_acc_mat["government"],
                        dev_acc_snli, dev_acc_mismat["nineeleven"], dev_acc_mismat["facetoface"],
                        dev_acc_mismat["letters"], dev_acc_mismat["oup"], dev_acc_mismat["verbatim"]))

                # Checkpoint
                if self.step % 500 == 0:
                    best_test = 100 * (1 - self.best_dev_tgenre_acc / dev_tgenre_acc)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_tgenre_acc = dev_tgenre_acc
                        self.best_train_sgenre_acc = train_sgenre_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best dev accuracy: %f" % (self.best_dev_tgenre_acc))

                if self.step % 2000 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    logger.Log("Checkpoint: %d" % self.step)

                self.step += 1

                # Adaptation param and learning rate schedule as described in the paper
                p = float(self.step) / float(self.total_batch * self.num_epochs)
                self.lmbda = 2. / (1. + np.exp(-self.lmbda_rate * p)) - 1

                # Compute average loss
                avg_cost += total_cost / (self.total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f\t Lambda: %f" %(self.epoch+1, avg_cost, self.lmbda))
            
            self.epoch += 1
            self.last_train_sgenre_acc[(self.epoch % 5) - 1] = train_sgenre_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_sgenre_acc) / (5 * min(self.last_train_sgenre_acc)) - 1)

            if (progress < 0.1) or (self.step > self.best_step + 10000):
                logger.Log("Early stopping! Epoch: {} Step: {}".format(self.epoch, self.step - 1))
                logger.Log("Best tgenre-dev accuracy: %s" % (self.best_dev_tgenre_acc))
                logger.Log("Best sgenre-train accuracy: %s" %(self.best_train_sgenre_acc))
                # In case we use early stopping
                if self.num_epochs == 0:
                    self.completed = True
                    self.sess.close()
                    break

            # Stop training
            if self.epoch == self.num_epochs:
                logger.Log("Best tgenre-dev accuracy: %s" % self.best_dev_tgenre_acc)
                logger.Log("Best sgenre-train accuracy: %s" % self.best_train_sgenre_acc)
                self.completed = True
                self.sess.close()
                break


    def classify(self, examples):
        # This classifies a list of examples
        if (test is True) or (self.completed is True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
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

        if (test is True) or (self.completed is True):
            self.sess.close()

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


if not test:
    if os.path.exists(acclogpath):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    acclog = open(acclogpath, append_write)
    acclog.write("step,sgenre_train_acc,tgenre_train_acc,travel_dev_acc,fiction_dev_acc,slate_dev_acc,telephone_dev_acc,government_dev_acc,snli_dev_acc,nineeleven_dev_acc,facetoface_dev_acc,letters_dev_acc,oup_dev_acc,verbatim_dev_acc\n")
    classifier.train(training_sgenre, training_tgenre, dev_matched, dev_mismatched, dev_snli, acclog)
    acclog.close()
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
  

  
  