# Libraries
import numpy as np

def vectorizer(tokens_list,model):
    """
    Function for assigning vectorial representation of words in vocabulary.
    If a word does not exist in vocabulary its assigned vector is filled with as many zeros as the vector's size. 

    Args:
        tokens_list (list [str]): list of tokens to vectorize
        model (KeyedVectors file [.kv]): KeyedVectors object

    Returns:
        vector (numpy.ndarray): assigned vector for given token
    """
    
    vector = np.zeros(model.vector_size)
    count_tokens_in_word2vec = 0
    for token in tokens_list:
        #Check if word exists in Word2Vec vocabulary
        if token in model.key_to_index:
            token2vector = model[token]
            vector = vector + token2vector
            count_tokens_in_word2vec += 1

    if count_tokens_in_word2vec != 0:
        vector = vector/count_tokens_in_word2vec
    return vector


def build_vector_matrix(tickets_data,model):
    """
    Method for building Data Frame of ticket's number and its corresponding vectors, based on its tokens.

    Args:
        tickets_data (pandas.DataFrame): consolidated database's pre-processed Data Frame (must contain 'tokens_list' column).
        model (KeyedVectors object): mapping of all terms in vocabulary and its corresponding vectors.

    Returns:
        vector_matrix (pandas.DataFrame): Data Frame of ticket's number and its corresponding vectors, based on its tokens.
    """
    tickets_data['vector'] = tickets_data['tokens_list'].apply(vectorizer, model = model)
    vector_matrix = tickets_data[['ticket','vector']]
    return vector_matrix