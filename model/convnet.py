import numpy as np
import pandas as pd
import gensim
from keras.layers import Dense, Input, Flatten, Dropout, Merge, Conv1D, MaxPooling1D, Embedding
from keras.optimizers import SGD, Adam
from keras.models import Model
from itertools import product
import keras.backend as K
import functools
from gensim.models.keyedvectors import KeyedVectors
import _config

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, predSet, compileSet, trainable=True, extra_conv=False):
    """
    This is a simple convnet

    Params
    ----------
    embeddings: array, pre-trained or custom embeddings
    max_sequence_length: int, max numb of words to consider
    num_words: int, length of embedding dictionary
    embedding_dim: int, default to 300
    labels_index: int, dimention of labels/output
    predSet: dict, settings for predictions (includes: activation for last 2 dense layers of the network)
    compileSet: dict, setting for compiling the model (includes: loss function(keras string/object), optimzier(keras string/obj), metrics(keras string/obj) )
    trainable: bool, default True, is embedding layer trainable?
    extra_conv: bool, default False, add extra convolution filters
    ----------
    return: instance of model
    """

    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [4,8,10]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=64, kernel_size=filter_size, activation='elu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)

    conv = Conv1D(filters=64, kernel_size=6, activation='elu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv==True:
        x = Dropout(0.5)(l_merge)
    else:
        # Original Y.Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(64, activation=predSet['activation_dense'])(x)
    x = Dropout(0.5)(x)

    preds = Dense(predSet['outdim'], activation=predSet['activation_pred'])(x)


    model = Model(sequence_input, preds)
    model.compile(loss=compileSet['loss'],
                  optimizer=compileSet['opti'],
                  metrics=[compileSet['metrics']])
    return model

def w_categorical_crossentropy(y_true, y_pred, weights):
    """
    This is a symmetrical cost function

    Params
    ----------
    y_true: array, ground truth values
    y_pred: array, predicted values
    weights: array, (nested) the penalty values to apply with vertical axis = true class and horizontal axis = predicted class
    ----------
    return: int, the weighted loss
    """
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    return cross_ent * final_mask


custom_loss = functools.partial(w_categorical_crossentropy, weights=_config.INIT_COST_WEIGHTS)
custom_loss.__name__ ='w_categorical_crossentropy'
