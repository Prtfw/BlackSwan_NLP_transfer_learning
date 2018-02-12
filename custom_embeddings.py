import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer
from keras.layers import Dense, Input, Flatten, Dropout, Merge, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import keras.backend as K

import functools
from itertools import product
import pickle


'''
default args
'''
CSVPATH = "./mixmax_cleaned_2.csv"
DEFAULT_EMBEDDING_PATH = "./GoogleNews-vectors-negative300.bin.gz"


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT=.4
LABEL = 'tier'
VALIDATION_SPLIT = .2
EMB_TRAIN_TASK = {'dfcolumn': 'tags', 'is_categorical': True, 'min_count': 20 }

INIT_COST_WEIGHTS = np.ones((3,3))
INIT_COST_WEIGHTS[1,0]=5
INIT_COST_WEIGHTS[2,0]=15
INIT_COST_WEIGHTS[2,1]=1
DEFAULT_EMBEDDING_INDEX = gensim.models.KeyedVectors.load_word2vec_format(DEFAULT_EMBEDDING_PATH, binary=True)


'''
functions
'''

def load_df(path=CSVPATH):
    EMB_TRAIN_TASK_LABELS = EMB_TRAIN_TASK['dfcolumn']
    df = pd.read_csv(CSVPATH)
    hasTags = df[df[EMB_TRAIN_TASK_LABELS].notnull()]
    dftagged = hasTags.tags.str.split('\s*,\s*', expand=True)\
                    .stack().str.get_dummies().sum(level=0)
    has_tags = hasTags[hasTags[EMB_TRAIN_TASK_LABELS].notnull()]
    tags = has_tags[EMB_TRAIN_TASK_LABELS].str.split('\s*,\s*', expand=True)\
                    .stack().str.get_dummies().sum(level=0)
    summary_stats = tags.values.sum(axis=0)
    ALL_TAGS = list(tags)
    KEEP_TAGS = [ALL_TAGS[ind] for ind, count in enumerate(summary_stats) if count > EMB_TRAIN_TASK['min_count']]
    # print(list(KEEP_TAGS))
    tags= tags[KEEP_TAGS]
    df = pd.concat([hasTags, tags], axis=1)
    return df, KEEP_TAGS

def get_vocab (df, col):
    tokenizer = RegexpTokenizer(r'\w+')
    NEW_COL_NAME = str(col)+'_tokens'
    df[col] = df[col].astype(str)
    df[col+'_tokens'] = df[col].apply(tokenizer.tokenize)
    all_words = [word for tokens in df[NEW_COL_NAME] for word in tokens]
    txtdict = (list(set(all_words)))
    sentence_lengths = [len(tokens) for tokens in df[col]]
    VOCAB = sorted(list(set(all_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))
    return VOCAB

def processEmbeddings(df, col, VOCAB, baseEmbeddingDict, isMultilabel, labelCols):
    df[col]= df[col].astype(str)
    VOCAB_SIZE = len(VOCAB)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df[col].tolist())
    sequences = tokenizer.texts_to_sequences(df[col].tolist())
    labels = []
    word_index = tokenizer.word_index  #need 2 return

    print('Found %s unique tokens.' % len(word_index))

    X_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
    for word,index in word_index.items():
        embedding_weights[index,:] = baseEmbeddingDict[word] if word in baseEmbeddingDict else np.random.rand(EMBEDDING_DIM)
    print(embedding_weights.shape)

    if isMultilabel:
        labels = df[labelCols].as_matrix()
    else:
        labels = to_categorical(np.asarray(labelCols))[:,1:]

    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    X_data = X_data[indices]
    labels = labels[indices]
    ogText = df[col]

    return word_index, embedding_weights, X_data, labels, ogText

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, predSet, compileSet, trainable=True, extra_conv=False):
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [6,8,10]

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

def data_split(x, y, VALIDATION_SPLIT):
    num_validation_samples = int(VALIDATION_SPLIT * x.shape[0])
    x_train = x[:-num_validation_samples]
    y_train = y[:-num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]
    return x_train, y_train, x_val, y_val


def save_embedding_weights(emb_name, embeddingModel):
    names = [weight.name for layer in embeddingModel.layers for weight in layer.weights]
    embeddingModelWeights = embeddingModel.get_weights()


    for name, weight in zip(names, embeddingModelWeights):
        print(name, weight.shape)

    trained_embeddings = embeddingModelWeights[0]

    print(trained_embeddings.shape)
    '''
    # diff check
    '''
    #trained_embeddings-embedding_weights:
    '''
        save trained weights
    '''
    pickle_out = open(emb_name,"wb")
    pickle.dump(trained_embeddings, pickle_out)
    pickle_out.close()
    return trained_embeddings

def load_embeddings(path):
    pickle_in = open(path,"rb")
    custom_embedding_weights_loaded=  pickle.load(pickle_in)
    return custom_embedding_weights_loaded


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    return cross_ent * final_mask


custom_loss = functools.partial(w_categorical_crossentropy, weights=INIT_COST_WEIGHTS)
custom_loss.__name__ ='w_categorical_crossentropy'
