PATHS = dict(
CSVPATH= "./mixmax_cleaned_2.csv"
DEFAULT_EMBEDDING_PATH = "./../GoogleNews-vectors-negative300.bin.gz"
)

EMBEDDINGS = dict(
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT=.4
DEFAULT_EMBEDDING_INDEX
)


EMB_TRAIN_TASK = {'dfcolumn': 'tags', 'is_categorical': True, 'min_count': 20 }

INIT_COST_WEIGHTS = np.ones((3,3))
INIT_COST_WEIGHTS[1,0]=5
INIT_COST_WEIGHTS[2,0]=15
INIT_COST_WEIGHTS[2,1]=1

INIT_COST_WEIGHTS
