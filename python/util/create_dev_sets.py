import json
import sys
import random

num_samples_mat = 2000
num_samples_mis = 1000

genres_mat = {'fiction': 0,
              'government': 1,
              'slate': 2,
              'telephone': 3,
              'travel': 4}

genres_mis = {'facetoface': 0,
              'oup': 1,
              'letters': 2,
              'nineeleven': 3,
              'verbatim': 4}


def split_train():
    datapath = sys.argv[1]
    training_mnli = "{}/multinli_0.9/multinli_0.9_train.jsonl".format(datapath)
    train_eval_mnli = "{}/multinli_0.9/multinli_0.9_train_eval.jsonl".format(datapath)
    dev_mat_eval_mnli = "{}/multinli_0.9/multinli_0.9_dev_eval_matched.jsonl".format(datapath)
    counts = [0] * 5

    data = []
    with open(training_mnli, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    random.seed(1)
    random.shuffle(data)

    ft = open(train_eval_mnli, 'w')
    fd = open(dev_mat_eval_mnli, 'w')
    i = 0
    f_fd_first_line = True
    f_ft_first_line = True
    for line in data:
        if counts[genres_mat[line['genre']]] < num_samples_mat:
            if f_fd_first_line:
                json.dump(line, fd)
                f_fd_first_line = False
            else:
                fd.write('\n')
                json.dump(line, fd)
            counts[genres_mat[line['genre']]] += 1
        else:
            if f_ft_first_line:
                json.dump(line, ft)
                f_ft_first_line = False
            else:
                ft.write('\n')
                json.dump(line, ft)
        i += 1
        if i % 10000 == 0:
            print "done %d lines" % i

    ft.close()
    fd.close()


def split_mismatched():
    datapath = sys.argv[1]
    dev_mis_mnli = "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(datapath)
    dev_mis1_eval_mnli = "{}/multinli_0.9/multinli_0.9_dev_eval_mismatched1.jsonl".format(datapath)
    dev_mis2_eval_mnli = "{}/multinli_0.9/multinli_0.9_dev_eval_mismatched2.jsonl".format(datapath)
    counts = [0] * 5

    data = []
    with open(dev_mis_mnli, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    random.seed(1)
    random.shuffle(data)

    ft = open(dev_mis1_eval_mnli, 'w')
    fd = open(dev_mis2_eval_mnli, 'w')
    i = 0
    f_fd_first_line = True
    f_ft_first_line = True
    for line in data:
        if counts[genres_mis[line['genre']]] < num_samples_mis:
            if f_fd_first_line:
                json.dump(line, fd)
                f_fd_first_line = False
            else:
                fd.write('\n')
                json.dump(line, fd)
            counts[genres_mis[line['genre']]] += 1
        else:
            if f_ft_first_line:
                json.dump(line, ft)
                f_ft_first_line = False
            else:
                ft.write('\n')
                json.dump(line, ft)
        i += 1
        if i % 10000 == 0:
            print "done %d lines" % i

    ft.close()
    fd.close()


def main():
    #split_train()
    split_mismatched()


if __name__ == '__main__':
    main()
