import pickle
import _config


def save_embedding_weights(emb_name, embeddingModel):
    '''
    This saves the trained embedding weights to file

    Params
    ----------
    emb_name: array, the file name into which to save the weights
    embeddingModel: keras model instance, the model with which we trained the embeddings
    ----------
    return: array, the embeddings that were saved as an array
    '''
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
