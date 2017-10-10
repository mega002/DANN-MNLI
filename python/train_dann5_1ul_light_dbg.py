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
from models.bilstm5_1ul_light import *
import util.parameters5 as params
from util.data_processing5 import *
from util.evaluate_dbg import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
acclogpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".acc.log"
dbglogpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".dbg.log"
snlidebugpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".snli.dbg"
logger = logger.Logger(logpath)

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")

# Loading labeled data
training_labeled = []
for genre in GENRE_MAP:
    if GENRE_MAP[genre] < 5:
        training_data = load_nli_data_genre(FIXED_PARAMETERS["training_mnli"], genre, snli=False)
        training_labeled.extend(training_data)
        logger.Log("Added genre %s to training set: %d labeled samples" % (genre, len(training_data)))

# Loading unlabeled data
training_unlabeled = []
alpha = FIXED_PARAMETERS["alpha"]
training_data = load_nli_data_genre(FIXED_PARAMETERS["training_snli"], "snli", snli=True)
beta = int(len(training_data) * alpha)
training_unlabeled.extend(random.sample(training_data, beta))
logger.Log("Loaded unlabeled data from SNLI to training set: %d unlabeled samples" % (len(training_unlabeled)))

# Shuffle loaded data
random.seed(5)
random.shuffle(training_labeled)
random.shuffle(training_unlabeled)

dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

logger.Log("Loading embeddings")

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_labeled, training_unlabeled, dev_snli,
                                                                      dev_matched, dev_mismatched, test_matched,
                                                                      test_mismatched, test_snli])
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


######################### MODEL #############################

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.starter_learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.ul_batch_size = int(self.batch_size / 5)
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.two_steps_train = FIXED_PARAMETERS["two_steps_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"]
        self.lmbda_rate = FIXED_PARAMETERS["lmbda_rate"]
        self.lmbda = 0.
        self.num_epoches = FIXED_PARAMETERS["num_epoches"]
        self.init_step = FIXED_PARAMETERS["init_step"]

        logger.Log("Building model")

        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim,
                             embeddings=loaded_embeddings, emb_train=self.emb_train)

        self.total_cost = tf.add(self.model.pred_cost, self.model.domain_cost)

        # Perform GD with Adam
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100000, 0.96)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_cost,
                                                                             global_step=self.global_step)

        # TF init
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
        s_indices = range(self.batch_size * idx, self.batch_size * (idx + 1))
        t_indices = range(self.ul_batch_size * idx, self.ul_batch_size * (idx + 1))
        s_premise_vectors, s_hypothesis_vectors, s_labels, s_genres, s_domains = \
                self.get_minibatch_samples(labeled_samples, s_indices)
        t_premise_vectors, t_hypothesis_vectors, t_labels, t_genres, t_domains = \
                self.get_minibatch_samples(unlabeled_samples, t_indices)
        premise_vectors = np.vstack((s_premise_vectors, t_premise_vectors))
        hypothesis_vectors = np.vstack((s_hypothesis_vectors, t_hypothesis_vectors))
        labels = s_labels + t_labels
        genres = s_genres + t_genres
        domains = s_domains + t_domains
        return premise_vectors, hypothesis_vectors, labels, genres, domains

    def train(self, training_labeled, training_unlabeled, dev_mat, dev_mismat, dev_snli, acclog, dbglog):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.completed = False
        self.step = 1
        self.step_es = 0
        self.epoch = 0
        self.epoch_es = 0
        labeled_batches = int(len(training_labeled) / self.batch_size)
        unlabeled_batches = int(len(training_unlabeled) / self.ul_batch_size)
        self.total_batches = min(labeled_batches, unlabeled_batches)

        self.best_dev_snli = 0.
        self.best_train_labeled_acc = 0.
        self.last_train_labeled_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore best-checkpoint if it exists
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                best_dev_mat, dev_cost_mat = evaluate_classifier_genre(
                    self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat = evaluate_classifier_genre(
                    self.classify, dev_mismat, self.batch_size)
                self.best_dev_snli, dev_cost_mat = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_train_labeled_acc, train_labeled_cost = evaluate_classifier(
                                                self.classify, training_labeled[0:5000], self.batch_size)

                logger.Log("Restored best matched-dev acc: %r\n Restored best mismatched-dev acc: %r\nRestored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f" % (
                    best_dev_mat, best_dev_mismat, self.best_dev_snli, self.best_train_labeled_acc))
                self.sess.close()
                self.sess = tf.Session()

            self.saver.restore(self.sess, ckpt_file)
            self.step = self.init_step
            self.epoch = int(self.init_step / self.total_batches)
            logger.Log("Model restored from file: %s Step: %d Epoch: %d" % (ckpt_file, self.step, self.epoch+1))

        ### Training loop
        logger.Log("Training...")
        while True:
            random.shuffle(training_labeled)
            random.shuffle(training_unlabeled)
            avg_cost = 0.
            
            # Loop over all batches in epoch
            for i in range(self.total_batches):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                                        minibatch_domains = self.get_minibatch(training_labeled, training_unlabeled, i)
                
                # Run the optimizer to take a gradient step
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
                    dev_acc_mat, dev_cost_mat, dev_dacc_mat, dev_dcost_mat = evaluate_classifiers_genre(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier_genre(
                        self.classify, dev_mismat, self.batch_size)
                    dev_acc_snli, dev_cost_snli, dev_dacc_snli, dev_dcost_snli = evaluate_classifiers(self.classify, dev_snli, self.batch_size)
                    train_labeled_acc, train_labeled_cost, train_labeled_dacc, train_labeled_dcost = evaluate_classifiers(
                        self.classify, training_labeled[0:5000], self.batch_size)

                    acclog.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        self.step, train_labeled_acc, dev_acc_mat["travel"], dev_acc_mat["fiction"], dev_acc_mat["slate"],
                        dev_acc_mat["telephone"], dev_acc_mat["government"],
                        dev_acc_snli, dev_acc_mismat["nineeleven"], dev_acc_mismat["facetoface"],
                        dev_acc_mismat["letters"], dev_acc_mismat["oup"], dev_acc_mismat["verbatim"]))

                    dbglog.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        self.step, train_labeled_acc, train_labeled_dacc,
                        dev_acc_mat["travel"], dev_dacc_mat["travel"],
                        dev_acc_mat["fiction"], dev_dacc_mat["fiction"],
                        dev_acc_mat["slate"], dev_dacc_mat["slate"],
                        dev_acc_mat["telephone"], dev_dacc_mat["telephone"],
                        dev_acc_mat["government"], dev_dacc_mat["government"],
                        dev_acc_snli, dev_dacc_snli))

                # Checkpoint
                if self.step % 500 == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                    best_test = 100 * (1 - self.best_dev_snli / dev_acc_snli)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_snli = dev_acc_snli
                        self.best_train_labeled_acc = train_labeled_acc
                        self.best_step = self.step
                        logger.Log("Checkpointing with new best dev accuracy: %f" % self.best_dev_snli)

                if self.step % 2000 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    logger.Log("Checkpoint: %d" % self.step)

                self.step += 1

                # Adaptation param and learning rate schedule as described in the paper
                if self.two_steps_train:
                    numerator = 0 if self.step_es == 0 else self.step - self.step_es
                    p = float(numerator) / float(self.total_batches * (self.num_epoches - self.epoch_es))
                    self.lmbda = 2. / (1. + np.exp(-self.lmbda_rate * p)) - 1
                else:
                    p = float(self.step) / float(self.total_batches * self.num_epoches)
                    self.lmbda = 2. / (1. + np.exp(-self.lmbda_rate * p)) - 1

                # Compute average loss
                avg_cost += total_cost / (self.total_batches * self.batch_size)

            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f\t Lambda: %f" % (self.epoch+1, avg_cost, self.lmbda))
            
            self.epoch += 1
            self.last_train_labeled_acc[(self.epoch % 5) - 1] = train_labeled_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_labeled_acc) / (5 * min(self.last_train_labeled_acc)) - 1)
            if (progress < 0.1) or (self.step > self.best_step + 10000):
                logger.Log("Early stopping! Epoch: {} Step: {}".format(self.epoch, self.step-1))
                logger.Log("Best snli-dev accuracy: %s" % self.best_dev_snli)
                logger.Log("MultiNLI train labeled accuracy: %s" % self.best_train_labeled_acc)

                # In case we use early stopping
                if self.num_epoches == 0:
                    self.completed = True
                    self.sess.close()
                    break

                # Set ES step point for second training stage
                if self.step_es == 0:
                    self.step_es = self.step-1
                    self.epoch_es = self.epoch
                    if self.two_steps_train:
                        self.sess.close()
                        self.sess = tf.Session()
                        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
                        self.saver.restore(self.sess, best_path)
                        logger.Log("Model restored from file: %s" % best_path)

            # Stop training
            if self.epoch == self.num_epoches:
                logger.Log("Best snli-dev accuracy: %s" % self.best_dev_snli)
                logger.Log("MultiNLI train labeled accuracy: %s" % self.best_train_labeled_acc)
                self.completed = True
                self.sess.close()
                break

    def classify(self, examples, dbg=False):
        # This classifies a list of examples
        if (test is True) or (self.completed is True):
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        dlogits = np.empty(6)
        genres = []
        for i in range(total_batch):
            indices = range(self.batch_size * i, self.batch_size * (i + 1))
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
                minibatch_domains = self.get_minibatch_samples(examples, indices)

            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels,
                                self.model.d: minibatch_domains,
                                self.model.keep_rate_ph: 1.0,
                                self.model.idx: 0,
                                self.model.train: False}
            genres += minibatch_genres
            if dbg:
                logit, cost, dlogit, dcost = self.sess.run([self.model.pred_logits, self.model.pred_cost, self.model.domain_logits, self.model.domain_cost], feed_dict)
                dlogits = np.vstack([dlogits, dlogit])
            else:
                logit, cost = self.sess.run([self.model.pred_logits, self.model.pred_cost], feed_dict)
            logits = np.vstack([logits, logit])

        if (test is True) or (self.completed is True):
            self.sess.close()

        #if dbg:
        #    exp_logits = np.exp(logits[1:])
        #    probs = exp_logits / np.sum(exp_logits, axis=1)[:,None]
        #    return genres, probs, cost

        if dbg:
            return genres, np.argmax(logits[1:], axis=1), cost, np.argmax(dlogits[1:], axis=1), dcost
        else:
            return genres, np.argmax(logits[1:], axis=1), cost


######################### EXECUTION #############################

classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

# While test-set isn't released, use dev-sets for testing
#test_matched = dev_matched
#test_mismatched = dev_mismatched

if not test:
    if os.path.exists(acclogpath):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    acclog = open(acclogpath, append_write)
    acclog.write("step,labeled_train_acc,travel_dev_acc,fiction_dev_acc,slate_dev_acc,telephone_dev_acc,government_dev_acc,snli_dev_acc,nineeleven_dev_acc,facetoface_dev_acc,letters_dev_acc,oup_dev_acc,verbatim_dev_acc\n")
    dbglog = open(dbglogpath, append_write)
    dbglog.write("step,labeled_train_acc,labeled_train_dacc,travel_dev_acc,travel_dev_dacc,fiction_dev_acc,fiction_dev_dacc,slate_dev_acc,slate_dev_dacc,telephone_dev_acc,telephone_dev_dacc,government_dev_acc,government_dev_dacc,snli_dev_acc,snli_dev_dacc\n")
    classifier.train(training_labeled, training_unlabeled, dev_matched, dev_mismatched, dev_snli, acclog, dbglog)
    acclog.close()
    dbglog.close()
    logger.Log("Test acc on matched multiNLI: %s" %(evaluate_classifier(classifier.classify,
                                                    test_matched, FIXED_PARAMETERS["batch_size"]))[0])

    logger.Log("Test acc on mismatched multiNLI: %s" %(evaluate_classifier(classifier.classify,
                                                       test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])

    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify,
                                        test_snli, FIXED_PARAMETERS["batch_size"]))[0])
    #snlidbg = open(snlidebugpath, 'w')
    #snlidbg.write("pairID\tpremise\thypothesis\tlabel\tEprob\tNprob\tCprob\tcorrect\n")
    #logger.Log("Dev acc on SNLI: %s" % (evaluate_classifier_dbg(classifier.classify, dev_snli,
    #                                                            FIXED_PARAMETERS["batch_size"], snlidbg))[0])
    #snlidbg.close()
else:
    logger.Log("Test acc on matched multiNLI: %s" %(evaluate_classifier(classifier.classify,
                                                    test_matched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on mismatched multiNLI: %s" %(evaluate_classifier(classifier.classify,
                                                       test_mismatched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify,
                                                            test_snli, FIXED_PARAMETERS["batch_size"])[0]))
    
    # Results by genre,
    logger.Log("Test acc on matched genres: %s" %(evaluate_classifier_genre(classifier.classify,
                                                  test_matched, FIXED_PARAMETERS["batch_size"])[0]))

    logger.Log("Test acc on mismatched genres: %s" %(evaluate_classifier_genre(classifier.classify,
                                                     test_mismatched, FIXED_PARAMETERS["batch_size"])[0]))
  

  
