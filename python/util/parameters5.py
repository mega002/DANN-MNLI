"""
The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.

Required arguements are,
model_type: which model you wish to train with. Valid model types: cbow, bilstm, and esim.
model_name: the name assigned to the model being trained, this will prefix the name of the logs and checkpoint files.
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model_name", type=str, help="Give model name, this will name logs and checkpoints made. For example cbow, esim_test etc.")

parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--ckptpath", type=str, default="../logs")
parser.add_argument("--logpath", type=str, default="../logs")

parser.add_argument("--emb_to_load", type=int, default=None, help="Number of embeddings to load. If None, all embeddings are loaded.")
parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--keep_rate", type=float, default=0.5, help="Keep rate for dropout in the model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (labeled data)")
parser.add_argument("--seq_length", type=int, default=50, help="Max sequence length")
parser.add_argument("--emb_train", action='store_true', help="Call if you want to make your word embeddings trainable.")
parser.add_argument("--two_steps_train", action='store_true', help="Call if you want 2-steps training")
parser.add_argument("--alpha", type=float, default=0., help="What percentage of SNLI data to use in training")
parser.add_argument("--lmbda_rate", type=float, default=2., help="Lambda increasing rate")
parser.add_argument("--num_epoches", type=int, default=0, help="Number of epoches for training, 0 for early stopping (default)")
parser.add_argument("--init_step", type=int, default=1, help="Step number (for execution restore)")

parser.add_argument("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")

args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "model_name": args.model_name,
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train_eval.jsonl".format(args.datapath),
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_eval_matched.jsonl".format(args.datapath),
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_eval_mismatched1.jsonl".format(args.datapath),
        "test_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        "test_mismatched": "{}/multinli_0.9/multinli_0.9_dev_eval_mismatched2.jsonl".format(args.datapath),
        #"training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl".format(args.datapath),
        #"dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        #"dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath),
        #"test_matched": "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl".format(args.datapath),
        #"test_mismatched": "{}/multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl".format(args.datapath),
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
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "emb_train": args.emb_train,
        "two_steps_train": args.two_steps_train,
        "alpha": args.alpha,
        "lmbda_rate": args.lmbda_rate,
        "num_epoches": args.num_epoches,
        "init_step": args.init_step,
    }

    return FIXED_PARAMETERS

def train_or_test():
    return args.test
