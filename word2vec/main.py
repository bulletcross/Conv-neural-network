import os
import sys
import numpy as np
import tensorflow as tf

from collections import Counter
import random

DATA = 'Dataset/text8'
VOCAB = 'Dataset/vocabulary.tsv'
VOCAB_SIZE = 50
BATCH_SIZE = 128

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
        index = 0
        for word, _ in vocab_word_list:
            dictionary[word] = index
            index = index + 1
            file.write(word + '\n')
        key_value_dict = dict(zip(dictionary.values(), dictionary.keys()))
    word_index = [dictionary[word] if word in dictionary else 0 for word in all_words]
    return word_index

def single_data_generator(word_index, ctx_window):
    for index, word in enumerate(word_index):
        ctx = random.randint(1, ctx_window)
        for ctx_word in word_index[max(0, index - ctx): index]:
            yield word, ctx_word
        for ctx_word in word_index[index + 1: index + ctx + 1]:
            yield word, ctx_word


def batch_data_generator(batch_size, word_index, ctx_window):
    generator_fn = single_data_generator(word_index, ctx_window)
    while True:
        word_batch = np.zeros(BATCH_SIZE, dtype = np.int32)
        ctx_batch = np.zeros([BATCH_SIZE, 1])
        # Fill in the batch from single data generator
        for index in range(batch_size):
            word_batch[index], ctx_batch[index] = next(generator_fn)
        yield word_batch, ctx_batch

def main():
    word_index = data_generator_initialization()
    gen = single_data_generator(word_index, 2)
    for i in range(5):
        w,c = next(gen)
        print(w, c)

if __name__ == "__main__":
    main()
