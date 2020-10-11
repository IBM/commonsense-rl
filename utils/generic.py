import io
import os
import sys
import re
import numpy as np
import torch
import networkx as nx
import logging
import gensim.downloader as api
from gensim.models import KeyedVectors
from textworld import Game
from textworld.logic import State, Rule, Proposition, Variable
from functools import lru_cache
from pathlib import Path

# Logging formatting
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level='INFO', stream=sys.stdout)
emb_dict = {}


def getUniqueFileHandler(results_filename, ext='.pkl', mode='wb'):
    index = ''
    while True:
        if not os.path.isfile(results_filename + index + ext):
            return io.open(results_filename + index + ext, mode)
        else:
            if index:
                index = '(' + str(int(index[1:-1]) + 1) + ')'  # Append 1 to number in brackets
            else:
                index = '(1)'
            pass  # Go and try create file again


def load_embeddings(emb_loc, emb_type, separator='[<>+]', limit=None):
    print("Loading " + emb_type + ' embeddings ...', end="")
    if emb_type in emb_dict:
        print(' from cache.')
        return emb_dict[emb_type]
    if emb_type == 'w2v':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'GoogleNews-vectors-negative300.bin', binary=True,
                                                     limit=limit)
    elif emb_type == 'fasttext':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'crawl-300d-2M.vec', binary=False, limit=limit)
    elif emb_type == 'glove':
        emb_model = api.load("glove-wiki-gigaword-300")
    elif emb_type == 'complex':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'complex.txt', binary=False, limit=limit)
    elif emb_type == 'conceptnet-55-ppmi':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'conceptnet-55-ppmi-en.txt', binary=False, limit=limit)
    elif emb_type == 'numberbatch':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'numberbatch-en-19.08.txt', binary=False, limit=limit)
    elif emb_type == 'transh':
        emb_model = KeyedVectors.load_word2vec_format(emb_loc + 'transh.txt', binary=False, limit=limit)
    elif emb_type == 'random':
        emb_model = KeyedVectors(300)
    else:
        types = re.split(separator,emb_type)
        if len(types) <=1:
            print('Invalid Embedding name: '+ emb_type)
            return None
        models = [load_embeddings(emb_loc, t, separator=separator, limit=limit) for t in types]
        if '+' in emb_type: # append both the embedding together
            emb_model = concatenate_embeddings(models)
        elif '<' in emb_type: # overwrite the elements of 1 with 2 embedding
            emb_model = combine_embeddings(models)
        elif '>' in emb_type: # overwrite the elements of 2 with 1 embedding
            emb_model = combine_embeddings(reversed(models))
        else:
            raise NotImplementedError
    add_util_vectors(emb_model, 'zeros')
    emb_dict[emb_type] = emb_model
    print("  ("+str(len(emb_model.vocab)) +" words) Done.")
    return emb_model


def add_util_vectors(embeddings, pad_type='zeros'):
    if 'unk' in embeddings:
        embeddings["<UNK>"] = embeddings["unk"].copy()
    else:
        embeddings["<UNK>"] = build_padding(pad_type, embeddings.vector_size)
    if not '<PAD>' in embeddings:
        embeddings["<PAD>"] = build_padding(pad_type, embeddings.vector_size)


def build_padding(padding, size):
    if padding == 'random':
        return np.random.rand(size)
    elif padding == 'zeros':
        return np.zeros(size)
    else:
        raise NotImplementedError


def combine_embeddings(models):
    emb_dict = {}
    for model in models:
        temp_dict = {k: model.vectors[v.index] for (k, v) in model.vocab.items()}
        emb_dict.update(temp_dict)
        # emb_dict = {**emb_dict,**temp_dict}
    emb_sorted = sorted(emb_dict.items(), key=lambda x: x[0])
    words = [item[0] for item in emb_sorted]
    vectors = np.row_stack([item[1] for item in emb_sorted])
    result = KeyedVectors(model.vector_size)
    result.add(words, vectors)
    return result


def concatenate_embeddings(models, padding='random'):
    aligned_models = align_models(models, padding=padding)
    words = aligned_models[0].index2word
    vectors = np.column_stack([emb.vectors for emb in aligned_models])

    ncols = sum([emb.vector_size for emb in models])
    assert vectors.shape == (len(words), ncols)

    vector_size = vectors.shape[1]
    result = KeyedVectors(vector_size)
    result.add(words, vectors)
    return result


def align_vocabulary(model, global_vocab, padding='random'):
    vocab = set(model.vocab.keys())
    diff = list(global_vocab - vocab)
    if not diff:
        return model

    padding = {k: build_padding(padding, model.vector_size) for k in diff}
    emb_dict = {k: model.vectors[v.index] for (k, v) in model.vocab.items()}
    emb_dict.update(padding)
    emb_sorted = sorted(emb_dict.items(), key=lambda x: x[0])
    words = [item[0] for item in emb_sorted]
    vectors = np.row_stack([item[1] for item in emb_sorted])

    result = KeyedVectors(model.vector_size)
    result.add(words, vectors)
    return result


def align_models(models, padding='random'):
    vocab_list = [set(emb.vocab.keys()) for emb in models if emb]
    global_vocab = set.union(*vocab_list)

    return [align_vocabulary(m, global_vocab, padding=padding) if m else None for m in models]


def load_multiple_embeddings(emb_loc, emb_types):
    emb_models = {}
    for current_emb_type in emb_types:
        if current_emb_type is not None:
            if current_emb_type in emb_models:
                continue
            emb_models[current_emb_type] = load_embeddings(emb_loc, current_emb_type)
    return emb_models


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_tensor(np_matrix, device=torch.device("cpu"), type='long'):
    if type == 'long':
        return torch.from_numpy(np_matrix).type(torch.long).to(device)
    elif type == 'float':
        return torch.from_numpy(np_matrix).type(torch.float).to(device)


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def masked_softmax(x, m=None, dim=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def masked_mean(x, m=None, dim=1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res


def masked_ave_aggregator(x, mask):
    """
    ave aggregator of the node embedding
    input: tensor: batch x num_nodes x dim
    mask: tensor: batch x num_nodes

    """
    mask_sum = torch.sum(mask, -1).unsqueeze(-1) # batch x 1
    mask = mask.unsqueeze(-1) # batch x num_nodes x 1
    mask = mask.expand(-1, -1, x.shape[-1]) # batch x num_nodes x dim
    masked_x = x * mask # batch x num_nodes x dim
    sum_masked_x = torch.sum(masked_x, 1)
    ave_masked_x = sum_masked_x / mask_sum
    return ave_masked_x


def escape_entities(entities):
    return {re.sub(r'\s+', '_', e.lower().strip()) for e in entities}

