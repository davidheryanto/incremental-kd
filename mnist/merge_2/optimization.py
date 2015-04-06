__author__ = 'davidheryanto'

import sys
sys.path.append('/home/david/Dropbox/Projects/self_paced')

from mlp import *

import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import time, random, os

def set_params(file_name, classifier):
    with open(file_name, 'rb') as in_file:
        new_params = cPickle.load(in_file)
    for current_p, new_p in zip(classifier.params, new_params):
        current_p.set_value(new_p.eval())

def should_continue_training(min_valid_error_epoch, epoch_num, diff=50, min_epoch=50):
    if epoch_num < min_epoch:
        return True
    else:
        return epoch_num - min_valid_error_epoch < diff

def train(temperature, lambda_teacher):
    temperature = np.float32(temperature)
    lambda_teacher = np.float32(lambda_teacher)

    learning_rate = 0.30
    L1_reg = 0.00
    L2_reg = 1e-4
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

    rng = numpy.random.RandomState(1234)
    random.seed(9999)

    model_1 = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperature,
    )
    set_params('./model_1_best.pkl', model_1)

    model_merged = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        temperature=temperature,
    )
    set_params('model_1_best.pkl', model_merged)

    cost_wrt_target = (
        -T.log(model_merged.p_y_given_x)[T.arange(y.shape[0]), y]
    )

    cost_wrt_teacher = (
        T.sum(-T.log(model_merged.p_y_given_x_relaxed) * model_1.p_y_given_x_relaxed,
              axis=1)
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

    # Train model
    valid_error = []
    min_valid_error = np.inf
    min_valid_error_epoch = 0
    minibatch_indexes = range(int(n_train_batches * 0.50), int(n_train_batches * 1.00))
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

        if epoch_num % 50 == 0:
            print(
                '|{:>4}|{:.4f}|{:.4f} ({:>4})|{:>4.1f}'.format(
                epoch_num, valid_loss, min_valid_error, min_valid_error_epoch,
                time.time() - start_time))


    return min_valid_error

def main(job_id, params):
    return train(temperature=params['temperature'][0],
                 lambda_teacher=params['lambda_teacher'][0],)


# if __name__ == '__main__':
#     train(7, 0.80)
