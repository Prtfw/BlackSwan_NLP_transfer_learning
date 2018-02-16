from gensim.models.keyedvectors import KeyedVectors
import numpy as np

PATHS = dict(
CSVPATH= "./data_cleaned.csv",
DEFAULT_EMBEDDING_PATH = "./GoogleNews-vectors-negative300.bin.gz",
EMB_PATH="./saved_outputs/custom_trained_embeddings1"
)

EMBEDDINGS = dict(
EMBEDDING_DIM = 300,
MAX_SEQUENCE_LENGTH = 1000,
VALIDATION_SPLIT=.4,
VOCAB_COL_NAMES = 'text',
DEFAULT_EMBEDDING_INDEX = KeyedVectors.load_word2vec_format(PATHS['DEFAULT_EMBEDDING_PATH'], binary=True)
)


EMB_TRAIN_TASK = {'dfcolumn': 'tags', 'is_categorical': True, 'min_count': 20 }

INIT_COST_WEIGHTS = np.ones((3,3))
INIT_COST_WEIGHTS[1,0]=5
INIT_COST_WEIGHTS[2,0]=15
INIT_COST_WEIGHTS[2,1]=1

INIT_COST_WEIGHTS
LABEL_COL = 'tier'
