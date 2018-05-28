import os
import sys
import numpy as np
import tensorflow as tf

from collections import Counter

DATA = 'Dataset/text8'
VOCAB = 'Dataset/vocabulary.tsv'
VOCAB_SIZE = 50

def data_generator_initialization():
    # compat = compatibility with python bytes and unicode
    with open(DATA, 'r') as file:
        all_words = tf.compat.as_str(file.read()).split()
    # Build and save vocabulary in tsv (tensorflow's csv)
    with open(VOCAB, 'w') as file:
        dictionary = dict() # Empty python dictionary
        vocab_word_list = [('UNK', -1)]
        vocab_index = 0
        # Include the most recurring words in list
        vocab_word_list.extend(Counter(all_words).most_common(VOCAB_SIZE-1))
        for word, _ in vocab_word_list:
            dictionary[word] = index
            index = index + 1
            file.write(word + '\n')
        key_value_dict = dict(zip(dictionary.values(), dictionary.keys()))
    word_index = [dictionary[word] if word in dictionary else 0 for word in all_words]
    return word_index
