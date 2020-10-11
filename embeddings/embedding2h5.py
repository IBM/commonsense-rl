# encoding: utf-8
import io
import numpy as np
import h5py


def export_data_h5(vocabulary, embedding_matrix, output='embedding.h5'):
    f = h5py.File(output, "w")
    compress_option = dict(compression="gzip", compression_opts=9, shuffle=True)
    words_flatten = '\n'.join(vocabulary)
    f.attrs['vocab_len'] = len(vocabulary)
    dt = h5py.special_dtype(vlen=str)
    _dset_vocab = f.create_dataset('words_flatten', (1, ), dtype=dt, **compress_option)
    _dset_vocab[...] = [words_flatten]
    _dset = f.create_dataset('embedding', embedding_matrix.shape, dtype=embedding_matrix.dtype, **compress_option)
    _dset[...] = embedding_matrix
    f.flush()
    f.close()


def fasttext_export(embedding_file):
    fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocabulary = []
    embeddings = []
    line_idx = 0
    for line in fin:
        if line_idx == 0:
            line_idx += 1
            continue
        tokens = line.rstrip().split(' ')
        vocabulary.append(tokens[0])
        embeddings.append([float(item) for item in tokens[1:]])
        line_idx += 1
    export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=embedding_file + ".h5")


if __name__ == '__main__':
    fasttext_export('crawl-300d-2M.vec')