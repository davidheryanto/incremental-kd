__author__ = 'davidheryanto'

import sys
sys.path.append('/home/david/Dropbox/Projects/self_paced')

from mlp import *

import os, cPickle, operator, time, random
from collections import defaultdict, OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from copy import deepcopy

from logistic_sgd import *
from mlp import *

def set_avg_params(path_0, path_1, model):
    with open(path_0, 'rb') as in_file:
        params_0 = cPickle.load(in_file)

    with open(path_1, 'rb') as in_file:
        params_1 = cPickle.load(in_file)

    avgs = []
    for p0, p1 in zip(params_0[4:], params_1[4:]):
        avgs.append( (p0.get_value() + p1.get_value())  / 2.0  )

    for current_p, new_p in zip(model.params[4:], avgs):
        current_p.set_value(new_p)


def should_continue_training(min_valid_error_epoch, epoch_num, diff=5, min_epoch=10):
    if epoch_num < min_epoch:
        return True
    else:
        return epoch_num - min_valid_error_epoch < diff


def train(temperature, lambda_teacher):
    temperature = np.float32(temperature)
    lambda_teacher = np.float32(lambda_teacher)

    learning_rate = 0.050
    L1_reg = 0
    L2_reg = 1e-4
    batch_size = 100

    nkerns = (80, 40)
    n_in = 500

    ######### LOAD DATA ###########
    # Use cifar10_david / cifar10_davidheryanto
    cifar_path = os.path.join(
        os.getenv('PYLEARN2_DATA_PATH'), 'cifar10_david'
    )
    datasets = load_cifar10(cifar_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(23455)
    random.seed(51665)

    model_1 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperature
    )
    model_2 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperature
    )

    # Set params
    set_params('./model_1_best.pkl', model_1)
    set_params('./model_2_best.pkl', model_2)

    model_merged = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperature
    )
    set_avg_params('model_1_best.pkl', 'model_2_best.pkl', model_merged)

    cost_wrt_target = (
        -T.log(model_merged.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_1 = (
        T.sum(-T.log(model_merged.p_y_given_x_relaxed) * model_1.p_y_given_x_relaxed,
              axis=1)
    )

    cost_wrt_model_2 = (
        T.sum(-T.log(model_merged.p_y_given_x_relaxed) * model_2.p_y_given_x_relaxed,
              axis=1)
    )

    cost_wrt_teacher = (
        ifelse(T.lt(index, n_train_batches * 0.5), cost_wrt_model_1, cost_wrt_model_2)
    )

    cost_model_merged = T.mean(
        (1 - lambda_teacher) * cost_wrt_target + lambda_teacher * cost_wrt_teacher
    )

    gparams_model_merged = (
        [T.grad(cost_model_merged, param) for param in model_merged.params]
    )

    validate_model_merged = theano.function(
        inputs=[index],
        outputs=model_merged.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_model_merged = theano.function(
        inputs=[index],
        outputs=model_merged.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates_model_merged = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model_merged.params, gparams_model_merged)
    ]

    train_model_merged = theano.function(
        inputs=[index],
        outputs=cost_model_merged,
        updates=updates_model_merged,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    ############# START THE ACTUAL TRAINING ################
    valid_error = []
    min_valid_error = np.inf
    min_valid_error_epoch = 0
    minibatch_indexes = range(int(n_train_batches * 0.25), int(n_train_batches * 0.75))
    epoch_num = 0
    print('|Epoch|Valid error|Min (Epoch)|Sec/epoch')
    print('|-:|-:|-:|-:')

    while should_continue_training(min_valid_error_epoch, epoch_num):
        epoch_num += 1
        start_time = time.time()
        random.shuffle(minibatch_indexes)

        for minibatch_index in minibatch_indexes:
            train_model_merged(minibatch_index)

        valid_loss = np.mean([validate_model_merged(i) for i in xrange(n_valid_batches)])
        valid_error.append(valid_loss)

        if valid_loss < min_valid_error:
            min_valid_error = valid_loss
            min_valid_error_epoch = epoch_num

        print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
            epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
            time.time() - start_time
        )

    return min_valid_error


def main(job_id, params):
    return train(temperature=params['temperature'][0],
                 lambda_teacher=params['lambda_teacher'][0])

# if __name__ == '__main__':
#     train(15, 0.20)