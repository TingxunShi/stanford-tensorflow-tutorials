from collections import Counter
import random
import os
import sys

import utils

sys.path.append('..')
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf


def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words


def build_vocab(words, vocab_size, visual_fld):
    """ Build vocabulary of VOCAB_SIZE most frequent words and write it to
    visualization/vocab.tsv
    """
    utils.safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')

    dictionary = dict()
    count = [('UNK', -1)]
    index = 0
    count.extend(Counter(words).most_common(vocab_size - 1))

    for word, _ in count:
        dictionary[word] = index
        index += 1
        file.write(word + '\n')

    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    file.close()
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_around_words(index_words, context_window_size):
    for index, center in enumerate(index_words[context_window_size: -context_window_size], start=context_window_size):
        context_words = index_words[index - context_window_size: index] + \
                        index_words[index + 1: index + context_window_size + 1]
        # assert len(context_words) == 2 * context_window_size
        yield (context_words, center)


def most_common_words(visual_fld, num_visualize):
    """ create a list of num_visualize most frequent words to visualize on TensorBoard.
    saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(visual_fld, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()


def batch_gen(download_url, expected_byte, vocab_size, batch_size,
              skip_window, visual_fld):
    local_dest = '../../examples/data/text8.zip'
    utils.download_one_file(download_url, local_dest, expected_byte)
    words = read_data(local_dest)
    dictionary, _ = build_vocab(words, vocab_size, visual_fld)
    index_words = convert_words_to_index(words, dictionary)
    del words  # to save memory
    single_gen = generate_around_words(index_words, skip_window)

    while True:
        around_batch = np.zeros([batch_size, 2 * skip_window], dtype=np.int32)
        center_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            around_batch[index], center_batch[index] = next(single_gen)
        yield around_batch, center_batch
