__author__ = 'davidheryanto'

from collections import OrderedDict
import operator
import numpy as np

from pylearn2.utils import serial
import theano

class sort_batch_by_confidence(object):
    def __init__(self, model_path, batch_size, min_confidence_percentile, debug=False):
        assert min_confidence_percentile >= 0
        assert min_confidence_percentile <= 1

        self.model = serial.load(model_path)
        self.batch_size = batch_size
        self.min_confidence_percentile = min_confidence_percentile
        self.debug = debug

    def apply(self, dataset, can_fit=False):
        assert len(dataset.X) == 50000

        # Theano function to get softmax output
        X = self.model.get_input_space().make_batch_theano()
        y = self.model.fprop(X)
        get_softmax = theano.function([X], y)

        n_train_batches = len(dataset.X) / self.batch_size
        batches_x = OrderedDict()
        batches_y = OrderedDict()
        for index in range(n_train_batches):
            batches_x[index] = dataset.X[index * self.batch_size : (index + 1) * self.batch_size]
            batches_y[index] = dataset.y[index * self.batch_size : (index + 1) * self.batch_size]

        confidence = OrderedDict()
        for index, batch in batches_x.items():
            confidence[index] = np.mean(np.max(get_softmax(batches_x[index]), axis=1))
        confidence_sorted = sorted(confidence.items(), key=operator.itemgetter(1))

        preprocessed_X = []
        preprocessed_y = []
        for (minibatch_index, confidence_val) in confidence_sorted:
            preprocessed_X.extend(batches_x[minibatch_index])
            preprocessed_y.extend(batches_y[minibatch_index])
        start_index = int(self.min_confidence_percentile * len(preprocessed_X))
        preprocessed_X = preprocessed_X[start_index:]
        preprocessed_y = preprocessed_y[start_index:]

        dataset.X = np.array(preprocessed_X, dtype=np.float32)
        dataset.y = np.array(preprocessed_y, dtype=np.uint8)

        if self.debug:
            print('=' * 80)
            print('len of X and y = ({}, {})'.format(len(dataset.X), len(dataset.y)))
            print('=' * 80)
