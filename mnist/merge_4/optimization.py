__author__ = 'davidheryanto'

import sys
sys.path.append('/home/david/Dropbox/Projects/self_paced')

from mlp import *
import numpy as np

import theano
import theano.tensor as T

import time, random, os

def set_params(file_name, classifier):
    with open(file_name, 'rb') as in_file:
        new_params = cPickle.load(in_file)
    for current_p, new_p in zip(classifier.params, new_params):
        current_p.set_value(new_p.eval())

def should_continue_training(min_valid_error_epoch, epoch_num, diff=100, min_epoch=500):
    max_epoch = 1000

    if epoch_num < min_epoch:
        return True
    elif epoch_num >= max_epoch:
        return False
    else:
        return epoch_num - min_valid_error_epoch < diff
    # if epoch_num < min_epoch:
    #     return True
    # else:
    #     return epoch_num - min_valid_error_epoch < diff

def train(temperatures, lambda_teachers):
    temperatures = np.asarray(temperatures, dtype=np.float32)
    lambda_teachers = np.asarray(lambda_teachers, dtype=np.float32)

    learning_rate = 0.30
    datapath = os.getenv('PYLEARN2_DATA_PATH')
    dataset = os.path.join(datapath, 'mnist.pkl.gz')
    batch_size = 100
    n_hidden = 250

    # Initialize data
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # Allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    """
    Stage 1
    """

    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_1 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )

    set_params('./model_1_best.pkl', model_1)

    model_merged_1 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )
    set_params('./model_1_best.pkl', model_merged_1)

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

    # Train model
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

        if epoch_num % 50 == 0:
            print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
                epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
                time.time() - start_time
            )

    """
    Stage 2
    """
    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_merged_1 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )

    set_params('./model_merged_1_best.pkl', model_merged_1)

    model_merged_2 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )
    set_params('./model_merged_1_best.pkl', model_merged_2)

    cost_wrt_target = (
        -T.log(model_merged_2.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_merged_1 = (
        T.sum(-T.log(model_merged_2.p_y_given_x_relaxed) * model_merged_1.p_y_given_x_relaxed,
              axis=1)
    )

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

        if epoch_num % 50 == 0:
            print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
                epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
                time.time() - start_time
            )

    """
    Stage 3
    """
    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_merged_2 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )

    set_params('./model_merged_2_best.pkl', model_merged_2)

    model_merged_3 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperatures,
    )
    set_params('./model_merged_2_best.pkl', model_merged_3)

    cost_wrt_target = (
        -T.log(model_merged_3.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_model_merged_2 = (
        T.sum(-T.log(model_merged_3.p_y_given_x_relaxed) * model_merged_2.p_y_given_x_relaxed,
              axis=1)
    )

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

    test_model_merged_3 = theano.function(
        inputs=[index],
        outputs=model_merged_3.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
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
    minibatch_indexes = range(int(n_train_batches * 0.75), int(n_train_batches * 1))
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
            save_params('model_merged_3_best.pkl', model_merged_3)

        if epoch_num % 50 == 0:
            print '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
                epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
                time.time() - start_time
            )

    return min_valid_error

def main(job_id, params):
    return train(temperatures=params['temperature'][0],
                 lambda_teachers=params['lambda_teacher'][0],)
