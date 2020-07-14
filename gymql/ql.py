import logging
import random
from collections import defaultdict

import keras
import numpy as np
import tflearn
from keras.layers import Dense, Input, Lambda, Add, BatchNormalization, Dropout, K
from keras.losses import mean_squared_error
from keras.models import Model
import tensorflow as tf
from keras.optimizers import Adam
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

GAMMA = 0.99
EPS = 1.0
MIN_EPS = 0.3
DECAY = 0.95
CHOOSE_FROM = 1

SAVE_TO = None  # os.path.dirname(os.path.realpath(__file__)) + '/q_model.weights'
LOAD_FROM = None  # os.path.dirname(os.path.realpath(__file__)) + '/q_model.weights'
SAMPLE_COUNT = 256

USE_DOUBLE = True
MAX_SMOOTHING = 4
USE_DUELING = True
BATCH_SIZE = 32
NORM_FACT = 1000

LAYER_PARAMS = {
    "activation": "relu",
    "use_bias": True,
    "kernel_initializer": "glorot_normal",
    "bias_initializer": "glorot_normal"
}

HIDDEN_STRUCTURE = [
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    },
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    },
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    }
]
SOFTMAX_TEMP = 0.8

DUELING_HIDDEN_LAYERS = [
    {
        "units": 128,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal"
    }
]

regularizers = {
    "l1": keras.regularizers.l1(0.001),
    "l2": keras.regularizers.l2(0.001)
}


class KerasQNetwork(object):
    def __init__(self,
                 num_inputs, num_outputs, hidden_structure=None,
                 use_dueling=USE_DUELING, dueling_hidden_layers=None, optimizer_args=None,
                 use_batch_norm=False, use_dropout=False, reg_target=None, reg_method=None):
        hidden_structure = hidden_structure or HIDDEN_STRUCTURE
        dueling_hidden_layers = dueling_hidden_layers or DUELING_HIDDEN_LAYERS
        optimizer_args = optimizer_args or {}
        input_layer = Input(shape=(num_inputs, ))

        additional_layer_args = {}
        if reg_target:
            additional_layer_args["{}_regularizer".format(reg_target)] = regularizers.get(reg_method, None)

        append_batchnorm = BatchNormalization() if use_batch_norm else lambda l: l
        append_dropout = Dropout(0.2) if use_dropout else lambda l: l

        def build_dense(layer, args):
            layer = Dense(**args, **additional_layer_args)(layer)
            layer = append_batchnorm(layer)
            return append_dropout(layer)

        def build_output_layer(num_o, input, name=None):
            return Dense(
                num_o,
                name=name,
                activation="linear",
                use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_normal"
            )(input)

        layer = input_layer
        for el in hidden_structure:
            layer = build_dense(layer, el)

        if use_dueling:
            state_layer = layer
            for el in dueling_hidden_layers:
                state_layer = build_dense(state_layer, el)
            state_layer = build_output_layer(1, state_layer)

            action_layer = layer
            for el in dueling_hidden_layers:
                action_layer = build_dense(action_layer, el)
            action_layer = build_output_layer(num_outputs, action_layer)

            norm_action = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keep_dims=True))(action_layer)
            layer = Add(name="output")([state_layer, norm_action])

        if not use_dueling:
            layer = build_output_layer(num_outputs, layer, "output")

        model = Model(inputs=input_layer, outputs=layer)
        model.compile(optimizer=Adam(**optimizer_args), loss=["mse"])
        self.model = model

    def fit_to(self, inputs, target_q, weights, masks):
        return self.model.train_on_batch(inputs, target_q), np.zeros((len(inputs),))

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions

    def copy_from(self, network):
        self.model.set_weights(network.model.get_weights())


TFLEARN_INITIALIZER = {
    "glorot_normal": "xavier"
}

TF_ADAM_PARAMS = {
    "lr": "learning_rate",
    "beta_1": "beta1",
    "beta_2": "beta2",
}

TFLEARN_LAYER_PARAMS = {
    "use_bias": "bias",
    "units": "n_units",
    "bias_initializer": "bias_init",
    "kernel_initializer": "weights_init"
}


def read_or_return(data, key):
    return data.get(key, key)


class TFQNetwork(object):
    def __init__(self,
                 num_inputs, num_outputs, hidden_structure=None,
                 use_dueling=USE_DUELING, dueling_hidden_layers=None, optimizer_args=None,
                 *args, **kwargs):
        self.session = tf.get_default_session()
        self.copy_network = None
        self.copy_ops = None
        hidden_structure = hidden_structure or HIDDEN_STRUCTURE
        dueling_hidden_layers = dueling_hidden_layers or DUELING_HIDDEN_LAYERS
        optimizer_args = optimizer_args or {}

        var_count_at_init = len(tf.trainable_variables())
        input_layer = tf.placeholder(tf.float32, shape=[None, num_inputs])
        self.input = input_layer

        # Current tensorflow implementation was only added to ensure that
        # no mistakes with the Keras PER implementation have been made
        # Because of this, the Low-Level implementation does only support Double, Dueling and PER
        # Especially Support for Dropout and Batch-Normalization are dropped for now

        def build_dense(layer, args):
            args = {read_or_return(TFLEARN_LAYER_PARAMS, k): v for k, v in args.items()}
            args["weights_init"] = read_or_return(TFLEARN_INITIALIZER, args["weights_init"])
            args["bias_init"] = read_or_return(TFLEARN_INITIALIZER, args["bias_init"])
            return tflearn.fully_connected(layer, **args)

        def build_output_layer(num_o, input):
            return tflearn.fully_connected(
                input,
                num_o,
                activation="linear",
                bias=True,
                weights_init="xavier",
                bias_init="xavier"
            )

        layer = input_layer
        for el in hidden_structure:
            layer = build_dense(layer, el)

        if use_dueling:
            state_layer = layer
            for el in dueling_hidden_layers:
                state_layer = build_dense(state_layer, el)
            state_layer = build_output_layer(1, state_layer)

            action_layer = layer
            for el in dueling_hidden_layers:
                action_layer = build_dense(action_layer, el)
            action_layer = build_output_layer(num_outputs, action_layer)

            norm_action = action_layer - tf.reduce_mean(action_layer, axis=1, keep_dims=True)
            layer = state_layer + norm_action
        else:
            layer = build_output_layer(num_outputs, layer)

        self.mask = tf.placeholder(tf.float32, [None, num_outputs])
        self.target = tf.placeholder(tf.float32, [None, num_outputs])
        self.loss_weights = tf.placeholder(tf.float32, [None, 1])
        self.output = layer
        self.loss_per_element = tf.reduce_mean(tf.square(self.target - layer) * self.mask, axis=1)
        self.loss = tf.reduce_mean(self.loss_weights * self.loss_per_element)

        optimizer_args = {read_or_return(TF_ADAM_PARAMS, k) : v for k, v in optimizer_args.items()}
        self.optimizer = tf.train.AdamOptimizer(**optimizer_args).minimize(self.loss)

        var_count_at_end = len(tf.trainable_variables())
        self.trainable_variables = tf.trainable_variables()[var_count_at_init:var_count_at_end]

    def fit_to(self, inputs, actual_q, weights, masks):
        ret_val = self.session.run([self.optimizer, self.loss, self.loss_per_element], feed_dict={
            self.input: inputs,
            self.target: actual_q,
            self.loss_weights: weights.reshape((-1, 1)),
            self.mask: masks
        })
        return ret_val[1:]

    def predict(self, inputs):
        return self.session.run([self.output], feed_dict={
            self.input: inputs.reshape((1, -1))
        })[0]

    def copy_from(self, network):
        ops = self.copy_ops
        if self.copy_network != network:
            ops = [tf.assign(input, output) for input, output in zip(network.trainable_variables, self.trainable_variables)]
            self.copy_network = network
            self.copy_ops = ops
        self.session.run(ops)


BACKENDS = {
    "tf": TFQNetwork,
    "keras": KerasQNetwork
}


class QLModel(object):

    def __init__(self, num_inputs, num_outputs,
                 log=None, eps=EPS, decay=DECAY, min_eps=MIN_EPS, network_params=None, use_double=USE_DOUBLE, gamma=GAMMA,
                 use_relative=False, network_backend="keras", *args, **kwargs):
        self.network_params = network_params or {}
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.use_double = use_double
        self.experience = defaultdict()

        self.iterations = 0
        self.update_ops = []
        self.network = None
        self.target = None
        self.log = log if log is not None else []
        self.network_backend = network_backend
        self.build_model()
        self.eps = eps
        self.gamma = gamma
        self.decay = decay
        self.min_eps = min_eps
        self.use_relative = use_relative

        self.losses = []

    def build_model(self):
        num_ins = self.num_inputs
        network_type = BACKENDS.get(self.network_backend, KerasQNetwork)

        self.network = network_type(num_ins, self.num_outputs, **self.network_params)
        if self.use_double:
            self.target = network_type(num_ins, self.num_outputs, **self.network_params)

        if self.network_backend == "tf":
            tf.get_default_session().run(tf.initialize_all_variables())

    def fit_to(self, *args, **kwargs):
        return self.network.fit_to(*args, **kwargs)

    def predict(self, inputs, use_target=False):
        network = self.target if use_target else self.network
        return network.predict(inputs)

    def predict_values(self, inputs, use_target=False):
        if np.any(np.isnan(inputs)):
            print("NaN in input")

        predictions = self.predict(inputs, use_target)

        predictions = predictions[0]
        if np.any(np.isnan(predictions)):
            print("NaN in output")

        return np.argmax(predictions), predictions

    def copy_to_target(self):
        self.target.copy_from(self.network)

    def train(self, tuples, weights):
        indexes = list(range(len(tuples)))
        random.shuffle(indexes)
        print('Training on the last ' + str(
            len(tuples)) + ' samples.')

        if self.use_double:
            self.copy_to_target()

        losses = []
        batch_x = np.zeros((BATCH_SIZE, self.num_inputs))
        batch_y = np.zeros((BATCH_SIZE, self.num_outputs))
        batch_weights = np.zeros((BATCH_SIZE,))
        batch_indexes = np.zeros((BATCH_SIZE,), dtype=np.int32)
        batch_masks = np.zeros((BATCH_SIZE, self.num_outputs))
        loss_updates = np.zeros(len(tuples))
        i = 0
        for idx in indexes:
            tup = tuples[idx]
            weight = weights[idx]

            # Find predicted values
            chose, predicted = self.predict_values(
                tup.s
            )

            # Find predictions for next step
            next_chose, _ = self.predict_values(
                tup.s2
            )
            _, next_predicted = self.predict_values(
                tup.s2,
                self.use_double
            )

            # Calculate at which output the price has actually been.
            pos = tup.a

            # Rewrite this value with the actual result value.
            predicted[pos] = self.normalize_reward(tup.r) + \
                             (self.gamma * next_predicted[next_chose])

            batch_masks[i][pos] = 1

            # If we have a relative action in the end and it was used, update the corresponding absolute action as well
            opp_relative = int((tup.s[0][1] + 0.5) * self.num_outputs) - 2
            if self.use_relative and (pos == self.num_outputs - 1) and opp_relative >= 0:
                predicted[opp_relative] = predicted[pos]
                batch_masks[i][opp_relative] = 1

            batch_x[i] = tup.s[0]
            batch_y[i] = predicted
            batch_weights[i] = weight
            batch_indexes[i] = idx

            i += 1

            if i == BATCH_SIZE:
                loss, loss_per_tuple = self.fit_to(batch_x, batch_y, batch_weights, batch_masks)
                losses.append(loss)
                for j, l in enumerate(loss_per_tuple):
                    loss_updates[batch_indexes[j]] = l
                i = 0

        if i > 0:
            i -= 1
            loss, loss_per_tuple = self.fit_to(batch_x[:i], batch_y[:i], batch_weights[:i], batch_masks[:i])
            losses.append(loss)
            for i, l in enumerate(loss_per_tuple):
                loss_updates[batch_indexes[i]] = l

        self.losses.append(sum(losses)/len(losses))

        self.eps *= self.decay
        if self.eps < self.min_eps:
            self.eps = self.min_eps

        return loss_updates

    def normalize_reward(self, r):
        return r / NORM_FACT

    def get_price(self, features):
        if np.random.uniform() < self.eps:
            self.log.append(False)
            return round(np.random.uniform(high=self.num_outputs - 1))
        else:
            self.log.append(True)
            predicted = self.predict_values(features)[0]
            return predicted
