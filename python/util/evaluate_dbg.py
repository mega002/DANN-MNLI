import numpy as np

def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1
    return correct / float(len(eval_set)), cost


def evaluate_classifiers(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    dcorrect = 0
    genres, hypotheses, cost, dhypotheses, dcost = classifier(eval_set, dbg=True)
    cost = cost / batch_size
    dcost = dcost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        dhypothesis = dhypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1
        if dhypothesis == eval_set[i]['domain']:
            dcorrect += 1
    return correct / float(len(eval_set)), cost, dcorrect / float(len(eval_set)), dcost


def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print 'welp!'

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost


def evaluate_classifiers_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost, dhypotheses, dcost = classifier(eval_set, dbg=True)
    correct = dict((genre, 0) for genre in set(genres))
    dcorrect = dict((genre, 0) for genre in set(genres))
    count = dict((genre, 0) for genre in set(genres))
    cost = cost / batch_size
    dcost = dcost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        dhypothesis = dhypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        if dhypothesis == eval_set[i]['domain']:
            dcorrect[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print 'welp!'

    accuracy = {k: correct[k] / count[k] for k in correct}
    daccuracy = {k: dcorrect[k] / count[k] for k in dcorrect}

    return accuracy, cost, daccuracy, dcost


def evaluate_classifier_dbg(classifier, eval_set, batch_size, snlidbg):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, logits, cost = classifier(eval_set, dbg=True)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        logit = logits[i]
        if np.argmax(logit) == eval_set[i]['label']:
            correct += 1
            snlidbg.write('{}\t"{}"\t"{}"\t{}\t{}\t{}\t{}\t{}\n'.format(eval_set[i]['pairID'], eval_set[i]['sentence1'],
                                                                eval_set[i]['sentence2'], eval_set[i]['label'],
                                                                logit[0], logit[1], logit[2], 'y'))
        else:
            snlidbg.write('{}\t"{}"\t"{}"\t{}\t{}\t{}\t{}\t{}\n'.format(eval_set[i]['pairID'], eval_set[i]['sentence1'],
                                                                eval_set[i]['sentence2'], eval_set[i]['label'],
                                                                logit[0], logit[1], logit[2], 'n'))
    return correct / float(len(eval_set)), cost