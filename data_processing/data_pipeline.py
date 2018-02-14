import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer



def load_df(path): #=CSVPATH
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


def data_split(x, y, VALIDATION_SPLIT):
    num_validation_samples = int(VALIDATION_SPLIT * x.shape[0])
    x_train = x[:-num_validation_samples]
    y_train = y[:-num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]
    return x_train, y_train, x_val, y_val
