"""
The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.

Required arguements are,
model_type: which model you wish to train with. Valid model types: cbow, bilstm, and esim.
model_name: the name assigned to the model being trained, this will prefix the name of the logs and checkpoint files.
"""

import argparse

parser = argparse.ArgumentParser()

# Valid genres to train on. 
genres = ['travel', 'fiction', 'slate', 'telephone', 'government']
def subtypes(s):
    options = [mod for mod in genres if s in genres]
    if len(options) == 1:
        return options[0]
    return s

parser.add_argument("model_name", type=str, help="Give model name, this will name logs and checkpoints made. For example cbow, esim_test etc.")

parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--ckptpath", type=str, default="../logs")
parser.add_argument("--logpath", type=str, default="../logs")

parser.add_argument("--emb_to_load", type=int, default=None, help="Number of embeddings to load. If None, all embeddings are loaded.")
parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--keep_rate", type=float, default=0.5, help="Keep rate for dropout in the model")
parser.add_argument("--seq_length", type=int, default=50, help="Max sequence length")
parser.add_argument("--emb_train", action='store_true', help="Call if you want to make your word embeddings trainable.")
parser.add_argument("--lmbda_init", type=float, default=0.1, help="Regularizing the domain cost")
parser.add_argument("--lmbda_rate", type=float, default=0.02, help="Increasing rate for domain cost regularization")

parser.add_argument("--sgenre", type=str, help="Which genre to train on")
parser.add_argument("--tgenre", type=str, help="Which genre to adapt to")
parser.add_argument("--alpha", type=float, default=0., help="What percentage of SNLI data to use in training")

parser.add_argument("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")

args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "model_name": args.model_name,
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl".format(args.datapath),
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath),
        "test_matched": "{}/multinli_0.9/multinli_0.9_test_matched.jsonl".format(args.datapath),
        "test_mismatched": "{}/multinli_0.9/multinli_0.9_test_mismatched.jsonl".format(args.datapath),
        "training_snli": "{}/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
        "dev_snli": "{}/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
        "test_snli": "{}/snli_1.0/snli_1.0_test.jsonl".format(args.datapath),
        "embedding_data_path": "{}/glove.840B.300d.txt".format(args.datapath),
        "log_path": "{}".format(args.logpath),
        "ckpt_path":  "{}".format(args.ckptpath),
        "embeddings_to_load": args.emb_to_load,
        "word_embedding_dim": 300,
        "hidden_embedding_dim": 300,
        "seq_length": args.seq_length,
        "keep_rate": args.keep_rate, 
        "batch_size": 32,
        "learning_rate": args.learning_rate,
        "emb_train": args.emb_train,
        "lmbda_init": args.lmbda_init,
        "lmbda_rate": args.lmbda_rate,
        "alpha": args.alpha,
        "source_genre": args.sgenre,
        "target_genre": args.tgenre
    }

    return FIXED_PARAMETERS

def train_or_test():
    return args.test

