import numpy as np
from gensim.models import KeyedVectors
import re


def save_dict_embeddings(path, embeddings, binary=None):
    if len(embeddings) == 0:
        raise ValueError("Empty embedding dictionary")

    if binary is None:
        if path.endswith('.txt'):
            binary = False
        else:
            binary = True

    gensim_model = dict_to_gensim(embeddings)
    gensim_model.save_word2vec_format(path, binary=binary)


def dict_to_gensim(embeddings):
    if len(embeddings) == 0:
        raise ValueError("Empty embedding dictionary")
    words = list(embeddings.keys())
    vectors = np.row_stack(list(embeddings.values()))
    result = KeyedVectors(vectors.shape[1])
    result.add(words, vectors)
    return result


def precompute_embeddings(model, vocabulary, path):
    result = {}
    for entity in vocabulary:
        tokens = re.split(r'\s|_', entity)
        entity_embedding = np.zeros(model.vector_size)
        count = 0
        for t in tokens:
            try:
                vector = model[t.lower()]
                entity_embedding += vector
                count += 1
            except KeyError:
                pass
        if count > 0:
            entity_embedding = entity_embedding / count
            entity = entity.strip().replace(' ', '_')
            result[entity] = entity_embedding
    if len(result) == 0:
        raise ValueError("No overlap between given model and vocabulary")
    save_dict_embeddings(path, result)
    return dict_to_gensim(result)
