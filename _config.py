from gensim.models.keyedvectors import KeyedVectors
import numpy as np

PATHS = dict(
CSVPATH= "./data_cleaned.csv", # path for load_df
DEFAULT_EMBEDDING_PATH = "./GoogleNews-vectors-negative300.bin.gz", # path for pre-trained embeddings
EMB_PATH="./saved_outputs/custom_trained_embeddings1" # path to save custom trained embeddings
)

EMBEDDINGS = dict(
EMBEDDING_DIM = 300,
MAX_SEQUENCE_LENGTH = 1000,
VALIDATION_SPLIT=.4,
VOCAB_COL_NAMES = 'text', # column name for text (input) feature
DEFAULT_EMBEDDING_INDEX = KeyedVectors.load_word2vec_format(PATHS['DEFAULT_EMBEDDING_PATH'], binary=True)
# dict to look up off the shelf embeddings
)


EMB_TRAIN_TASK = {'dfcolumn': 'tags', 'is_categorical': True, 'min_count': 20 }
# dfcolumn = label for embdding training task, is_categorical: are labels discrete, min_count: remove rare labels, set to 1 if continuous


INIT_COST_WEIGHTS = np.ones((3,3)) # intialize asymetrical penalty array for custom loss function (3 classes by default)
INIT_COST_WEIGHTS[1,0]=5 # penalty for misclassifying true class 2 as class 1
INIT_COST_WEIGHTS[2,0]=15
INIT_COST_WEIGHTS[2,1]=1


LABEL_COL = 'tier' # target/label column for 2nd stage task (for 2nd stage model after training embedding)
