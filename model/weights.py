import pickle


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
