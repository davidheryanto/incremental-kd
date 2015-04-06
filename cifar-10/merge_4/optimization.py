__author__ = 'davidheryanto'

import sys
sys.path.append('/home/david/Dropbox/Projects/self_paced')

import os, cPickle, operator, time, random
from collections import defaultdict, OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import copy
from copy import deepcopy

from mlp import *
from logistic_sgd import load_cifar10

def set_params(file_name, model):
    with open(file_name, 'rb') as in_file:
        new_params = cPickle.load(in_file)
    for current_p, new_p in zip(model.params, new_params):
        current_p.set_value(new_p.eval())

def should_continue_training(min_valid_error_epoch, epoch_num, diff=7, min_epoch=20):
    if epoch_num < min_epoch:
        return True
    else:
        return epoch_num - min_valid_error_epoch < diff

def set_avg_params(paths, model, layer_start=None, layer_end=None):
    if layer_start is None:
        layer_start = 0
    if layer_end is None:
        layer_end = len(model.params)

    params = []

    for path in paths:
        with open(path, 'rb') as in_file:
            params_loaded = cPickle.load(in_file)
            params.append(params_loaded[layer_start:layer_end])

    new_params = []
    for params_tuple in zip(*params):
        total = params_tuple[0].get_value()
        for param in params_tuple[1:]:
            total += param.get_value()
        avg = total / float(len(params))
        new_params.append(avg)

    for current_param, new_param in zip(model.params[layer_start:layer_end], new_params):
        # print(current_param.shape.eval(), new_param.shape)
        current_param.set_value(new_param)

def train(temperatures, lambda_teachers):
    temperatures = np.asarray(temperatures, dtype=np.float32)
    lambda_teachers = np.asarray(lambda_teachers, dtype=np.float32)

    learning_rate = 0.050
    L1_reg = 0
    L2_reg = 1e-4
    batch_size = 100

    nkerns = (80, 40)
    n_in = 500

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

    """
    Stage 1
    """
    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_merged_1 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_1_best.pkl'], model_merged_1, layer_start=4)

    model_1 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_1_best.pkl'], model_1)

    cost_wrt_target = (
        -T.log(model_merged_1.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_1 = (
        T.sum(-T.log(model_merged_1.p_y_given_x_relaxed) * model_1.p_y_given_x_relaxed,
              axis=1)
    )

    cost_wrt_teacher = cost_wrt_model_1

    cost_model_merged_1 = T.mean(
        (1 - lambda_teachers) * cost_wrt_target + lambda_teachers * cost_wrt_teacher
    )

    gparams_model_merged_1 = (
        [T.grad(cost_model_merged_1, param) for param in model_merged_1.params]
    )

    validate_model_merged_1 = theano.function(
        inputs=[index],
        outputs=model_merged_1.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates_model_merged_1 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model_merged_1.params, gparams_model_merged_1)
    ]

    train_model_merged_1 = theano.function(
        inputs=[index],
        outputs=cost_model_merged_1,
        updates=updates_model_merged_1,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_error = []
    min_valid_error = np.inf
    min_valid_error_epoch = 0
    minibatch_indexes = range(int(n_train_batches * 0.25), int(n_train_batches * 0.50))
    epoch_num = 0
    print('|Epoch|Valid error|Min (Epoch)|Sec/epoch')
    print('|-:|-:|-:|-:')

    while should_continue_training(min_valid_error_epoch, epoch_num):
        epoch_num += 1
        start_time = time.time()
        random.shuffle(minibatch_indexes)

        for minibatch_index in minibatch_indexes:
            train_model_merged_1(minibatch_index)

        valid_loss = np.mean([validate_model_merged_1(i) for i in xrange(n_valid_batches)])
        valid_error.append(valid_loss)

        if valid_loss < min_valid_error:
            min_valid_error = valid_loss
            min_valid_error_epoch = epoch_num

        print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
            epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
            time.time() - start_time
        )


    """
    Stage 2
    """
    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_merged_2 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_merged_1_best.pkl'], model_merged_2, layer_start=4)

    model_merged_1 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_merged_1_best.pkl'], model_merged_1)

    cost_wrt_target = (
        -T.log(model_merged_2.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_merged_1 = (
        T.sum(-T.log(model_merged_2.p_y_given_x_relaxed) * model_merged_1.p_y_given_x_relaxed,
              axis=1)
    )

    # cost_wrt_teacher = ifelse(T.lt(index, int(n_train_batches * 0.25)),
    #                           cost_wrt_model_1, cost_wrt_model_2)
    cost_wrt_teacher = cost_wrt_model_merged_1

    cost_model_merged_2 = T.mean(
        (1 - lambda_teachers) * cost_wrt_target + lambda_teachers * cost_wrt_teacher
    )

    gparams_model_merged_2 = (
        [T.grad(cost_model_merged_2, param) for param in model_merged_2.params]
    )

    validate_model_merged_2 = theano.function(
        inputs=[index],
        outputs=model_merged_2.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates_model_merged_2 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model_merged_2.params, gparams_model_merged_2)
    ]

    train_model_merged_2 = theano.function(
        inputs=[index],
        outputs=cost_model_merged_2,
        updates=updates_model_merged_2,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    min_valid_error_epoch = 0
    minibatch_indexes = range(int(n_train_batches * 0.50), int(n_train_batches * 0.75))
    epoch_num = 0
    print('|Epoch|Valid error|Min (Epoch)|Sec/epoch')
    print('|-:|-:|-:|-:')

    while should_continue_training(min_valid_error_epoch, epoch_num):
        epoch_num += 1
        start_time = time.time()
        random.shuffle(minibatch_indexes)

        for minibatch_index in minibatch_indexes:
            train_model_merged_2(minibatch_index)

        valid_loss = np.mean([validate_model_merged_2(i) for i in xrange(n_valid_batches)])
        valid_error.append(valid_loss)

        if valid_loss < min_valid_error:
            min_valid_error = valid_loss
            min_valid_error_epoch = epoch_num

        print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
            epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
            time.time() - start_time
        )

    """
    Stage 3
    """
    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_merged_3 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_merged_2_best.pkl'], model_merged_3, layer_start=4)

    model_merged_2 = MLPConv(
        rng=rng,
        model_input=x,
        nkerns=nkerns,
        n_in=n_in,
        temperature=temperatures
    )
    set_avg_params(['model_merged_2_best.pkl'], model_merged_2)

    cost_wrt_target = (
        -T.log(model_merged_3.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_merged_2 = (
        T.sum(-T.log(model_merged_3.p_y_given_x_relaxed) * model_merged_2.p_y_given_x_relaxed,
              axis=1)
    )

    # cost_wrt_teacher = ifelse(T.lt(index, int(n_train_batches * 0.25)),
    #                           cost_wrt_model_1, cost_wrt_model_2)
    cost_wrt_teacher = cost_wrt_model_merged_2

    cost_model_merged_3 = T.mean(
        (1 - lambda_teachers) * cost_wrt_target + lambda_teachers * cost_wrt_teacher
    )

    gparams_model_merged_3 = (
        [T.grad(cost_model_merged_3, param) for param in model_merged_3.params]
    )

    validate_model_merged_3 = theano.function(
        inputs=[index],
        outputs=model_merged_3.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    updates_model_merged_3 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model_merged_3.params, gparams_model_merged_3)
    ]

    train_model_merged_3 = theano.function(
        inputs=[index],
        outputs=cost_model_merged_3,
        updates=updates_model_merged_3,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    min_valid_error_epoch = 0
    minibatch_indexes = range(int(n_train_batches * 0.75), int(n_train_batches * 1.00))
    epoch_num = 0
    print('|Epoch|Valid error|Min (Epoch)|Sec/epoch')
    print('|-:|-:|-:|-:')

    while should_continue_training(min_valid_error_epoch, epoch_num):
        epoch_num += 1
        start_time = time.time()
        random.shuffle(minibatch_indexes)

        for minibatch_index in minibatch_indexes:
            train_model_merged_3(minibatch_index)

        valid_loss = np.mean([validate_model_merged_3(i) for i in xrange(n_valid_batches)])
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
    return train(temperatures=params['temperature'][0],
                 lambda_teachers=params['lambda_teacher'][0],)