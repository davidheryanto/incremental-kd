"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import cPickle

import numpy
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression, load_data


def save_params(file_name, classifier):
    with open(file_name, 'wb') as out_file:
        cPickle.dump(classifier.params, out_file)


def set_params(file_name, classifier):
    with open(file_name, 'rb') as in_file:
        new_params = cPickle.load(in_file)
    for current_p, new_p in zip(classifier.params, new_params):
        current_p.set_value(new_p.eval())


def reset_params(file_name, classifier):
    with open(file_name, 'rb') as in_file:
        initial_params = cPickle.load(in_file)

    print('Before reset:')
    for p in classifier.params[:2]:
        print(p[0].eval().ravel()[:5])

    for initial, current in zip(initial_params, classifier.params):
        current.set_value(initial.eval())

    print('After reset')
    for p in classifier.params[:2]:
        print(p[0].eval().ravel()[:5])


def save_valid_error(file_name, valid_error):
    with open(file_name, 'wb') as out_file:
        cPickle.dump(valid_error, out_file)


def load_valid_error(file_name):
    with open(file_name, 'rb') as in_file:
        return cPickle.load(in_file)


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        # activation function used (among other things).
        # For example, results presented in [Xavier10] suggest that you
        # should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_p, W=None, b=None,
                 activation=T.tanh):
        """
        :param rng: Numpy random number generator
        :type rng: numpy.random.RandomState
        """
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, activation=activation
        )
        self.output = _dropout_from_layer(rng, self.output, p=dropout_p)


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        from theano.tensor.signal import downsample
        from theano.tensor.nnet import conv

        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        if W is None:
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class DropoutLenetConvPoolLayer(LeNetConvPoolLayer):
    def __init__(self, rng, input, filter_shape, image_shape, dropout_p, poolsize=(2, 2)):
        super(DropoutLenetConvPoolLayer, self).__init__(
            rng=rng, input=input, filter_shape=filter_shape, image_shape=image_shape, poolsize=poolsize
        )
        self.output = _dropout_from_layer(rng, self.output, p=dropout_p)


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    SEED = 3912309
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(SEED))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, dropout_ps, temperature=1):
        """Initialize the parameters for the multilayer perceptron

        :type dropout_ps: list
        :param dropout_ps: probability of dropping each activation units in
                           each layer, starting from the input layer

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_ps[0])

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.dropoutHiddenLayer = DropoutHiddenLayer(
            rng=rng,
            input=next_dropout_layer_input,
            n_in=n_in,
            n_out=n_hidden,
            dropout_p=dropout_ps[1],
            activation=T.tanh
        )

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=next_layer_input,
            activation=T.tanh,
            W=self.dropoutHiddenLayer.W * (1 - dropout_ps[0]),
            b=self.dropoutHiddenLayer.b,
            n_in=n_in,
            n_out=n_out
        )

        self.dropoutLogRegressionLayer = LogisticRegression(
            input=self.dropoutHiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            temperature=temperature
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            W=self.dropoutLogRegressionLayer.W * (1 - dropout_ps[1]),
            b=self.dropoutLogRegressionLayer.b,
            n_in=n_hidden,
            n_out=n_out,
            temperature=temperature
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.dropoutHiddenLayer.W).sum()
            + abs(self.dropoutLogRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.dropoutHiddenLayer.W ** 2).sum()
            + (self.dropoutLogRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layera
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.dropout_negative_log_likelihood = self.dropoutLogRegressionLayer.negative_log_likelihood

        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.p_y_given_x_relaxed = self.logRegressionLayer.p_y_given_x_relaxed

        self.dropout_p_y_given_x = self.dropoutLogRegressionLayer.p_y_given_x
        self.dropout_p_y_given_x_relaxed = self.dropoutLogRegressionLayer.p_y_given_x_relaxed

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.droput_errors = self.dropoutLogRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.dropoutHiddenLayer.params + self.dropoutLogRegressionLayer.params


class MLPConv(object):
    def __init__(self,
                 rng,
                 model_input,
                 image_shape=(3, 32, 32),
                 filter_shape=(5, 5),
                 poolsize=(2, 2),
                 batch_size=100,
                 nkerns=(20, 50),
                 n_in=400,
                 n_out=10,
                 temperature=1,
                 dropout_ps=[0.0, 0.0, 0.0, 0.0]):
        layer0_input = model_input.reshape((batch_size,) + image_shape)
        layer0_input_dropout = _dropout_from_layer(rng, layer0_input, p=dropout_ps[0])

        self.layer0_dropout = DropoutLenetConvPoolLayer(
            rng,
            input=layer0_input_dropout,
            image_shape=(batch_size,) + image_shape,
            filter_shape=(nkerns[0],) + (image_shape[0],) + filter_shape,
            poolsize=poolsize,
            dropout_p=dropout_ps[1]
        )
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            W=self.layer0_dropout.W * (1 - dropout_ps[0]),
            b=self.layer0_dropout.b,
            image_shape=(batch_size,) + image_shape,
            filter_shape=(nkerns[0],) + (image_shape[0],) + filter_shape,
            poolsize=poolsize,
        )

        self.layer1_dropout = DropoutLenetConvPoolLayer(
            rng,
            input=self.layer0_dropout.output,
            image_shape=(batch_size,) + (nkerns[0],) + (14, 14),
            filter_shape=(nkerns[1], nkerns[0]) + (5, 5),
            poolsize=poolsize,
            dropout_p=dropout_ps[2],
        )
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            W=self.layer1_dropout.W * (1 - dropout_ps[1]),
            b=self.layer1_dropout.b,
            image_shape=(batch_size,) + (nkerns[0],) + (14, 14),
            filter_shape=(nkerns[1], nkerns[0]) + (5, 5),
            poolsize=poolsize,
        )

        self.layer2_dropout = DropoutHiddenLayer(
            rng,
            input=self.layer1_dropout.output.flatten(2),
            n_in=nkerns[1] * 5 * 5,
            n_out=n_in,
            activation=T.tanh,
            dropout_p=dropout_ps[3],
        )
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer1.output.flatten(2),
            W=self.layer2_dropout.W * (1 - dropout_ps[2]),
            b=self.layer2_dropout.b,
            n_in=nkerns[1] * 5 * 5,
            n_out=n_in,
            activation=T.tanh,
        )

        self.logRegressionLayer_dropout = LogisticRegression(
            input=self.layer2_dropout.output,
            n_in=n_in,
            n_out=n_out,
            temperature=temperature
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.layer2.output,
            W=self.logRegressionLayer_dropout.W * (1 - dropout_ps[3]),
            b=self.logRegressionLayer_dropout.b,
            n_in=n_in,
            n_out=n_out,
            temperature=temperature
        )

        # self.L1 = (
        #     abs(self.layer1_dropout.W).sum()
        #     + abs(self.layer2_dropout.W).sum()
        #     + abs(self.logRegressionLayer.W_dropout).sum()
        # )
        #
        # self.L2_sqr = (self.layer1.W ** 2).sum() + \
        #               (self.layer2.W ** 2).sum() + \
        #               (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.negative_log_likelihood_dropout = self.logRegressionLayer_dropout.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.errors_dropout = self.logRegressionLayer_dropout.errors

        self.params = (
            self.logRegressionLayer_dropout.params
            + self.layer2_dropout.params
            + self.layer1_dropout.params
            + self.layer0_dropout.params
        )

        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.p_y_given_x_relaxed = self.logRegressionLayer.p_y_given_x_relaxed


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', drouput_ps=[0.2, 0.5], batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    data_path = os.getenv('DATA_PATH', '.')
    datasets = load_data(os.path.join(data_path, 'mnist.pkl.gz'))

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        dropout_ps=drouput_ps
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.dropout_negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    # C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
