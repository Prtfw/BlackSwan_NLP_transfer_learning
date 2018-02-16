import tensorflow as tf
import _config as _config
import numpy as np
from model import weights, convnet
from data_processing import data_pipeline
from keras.optimizers import SGD, Adam
from sklearn.utils import class_weight
from nltk.tokenize import RegexpTokenizer
from keras.models import load_model

if __name__ == "__main__":
    # process data for embedding training
    df, KEEP_TAGS  = data_pipeline.load_df(_config.PATHS['CSVPATH'])
    TEST_VOCAB = data_pipeline.get_vocab(df, _config.EMBEDDINGS['VOCAB_COL_NAMES'])
    custom_word_index, init_embedding_weights,  x_emb, labels_emb, ogText= data_pipeline.process_embeddings(
        df,
        _config.EMBEDDINGS['VOCAB_COL_NAMES'],
        TEST_VOCAB, _config.EMBEDDINGS['DEFAULT_EMBEDDING_INDEX'],
        True,
        KEEP_TAGS)
    x_train_emb, y_train_emb, x_val_emb, y_val_emb = data_pipeline.data_split(x_emb,labels_emb,
        _config.EMBEDDINGS['VALIDATION_SPLIT'])

    # define embedding training model
    embeddingPred = {'outdim': y_train_emb.shape[1], 'activation_pred': 'sigmoid', 'activation_dense': 'elu'}
    embeddingCompile = {'loss': 'binary_crossentropy', 'opti': 'adam', 'metrics': 'categorical_accuracy'}
    embeddingModel = convnet.ConvNet(init_embedding_weights, _config.EMBEDDINGS['MAX_SEQUENCE_LENGTH'],
        len(custom_word_index)+1, _config.EMBEDDINGS['EMBEDDING_DIM'], len(KEEP_TAGS),
            embeddingPred, embeddingCompile)

    '''
    # uncomment following block to train and save custom embdding weights
    '''
    # embeddingModel.fit(x_train_emb, y_train_emb, validation_data=(x_val_emb, y_val_emb), epochs=1, batch_size=32)
    # trained_emb = weights.save_embedding_weights(_config.PATHS['EMB_PATH'], embeddingModel)


    '''
    # uncomment to load saved custom embeddings
    '''
    trained_emb = weights.load_embeddings(_config.PATHS['EMB_PATH'])
    print('check diff', init_embedding_weights-trained_emb)

    # process data for classifier training
    df_clf = df[(df[_config.EMB_TRAIN_TASK['dfcolumn']].notnull()) & (df[_config.LABEL_COL].notnull())]
    df_clf[_config.LABEL_COL] = df_clf[_config.LABEL_COL].astype(int)
    CLF_VOCAB = data_pipeline.get_vocab(df_clf, _config.EMBEDDINGS['VOCAB_COL_NAMES'])
    clf_word_index, clf_embedding_weights,  x_clf, labels_clf, ogText= data_pipeline.process_embeddings(
        df_clf,
        _config.EMBEDDINGS['VOCAB_COL_NAMES'],
        CLF_VOCAB,
        custom_word_index,
        False,
        df_clf[_config.LABEL_COL])
    x_train_clf, y_train_clf, x_val_clf, y_val_clf = data_pipeline.data_split(x_clf,labels_clf,
        _config.EMBEDDINGS['VALIDATION_SPLIT'])
    y_train_class = [list(arr) for arr in y_val_clf]
    y_train_class = [list(arr).index(max(list(arr))) for arr in y_train_class]
    y_train_class
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train_class), y_train_class)
    print(class_weight)

    # define classifier training model
    learning_rate = 0.001
    decay_rate = learning_rate / 50
    momentum = 0.9
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=True)
    adam = Adam(lr=0.01)
    clfPred = {'outdim':3, 'activation_pred': 'sigmoid', 'activation_dense': 'sigmoid'}
    clfCompile = {'loss': convnet.custom_loss,
        'opti': adam, 'metrics': 'acc'}
    clfModel = convnet.ConvNet(clf_embedding_weights, 1000, len(clf_word_index)+1, 300, 3,
        clfPred, clfCompile, trainable=False, extra_conv=True)

    clfModel.fit(x_train_clf, y_train_clf, validation_data=(x_val_clf, y_val_clf),
             epochs=1, batch_size=32, class_weight=class_weight )
    clfModel.save('./saved_outputs/clfModel.h5')
    print('model saved')
