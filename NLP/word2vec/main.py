import os
import sys
import numpy as np
import tensorflow as tf

from collections import Counter
import random

DATA = 'Dataset/text8'
VOCAB = 'Dataset/vocabulary.tsv'
VOCAB_SIZE = 10000
EMBED_SIZE = 50
BATCH_SIZE = 128
NR_STEPS = 10000

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

def generator():
    word_index = data_generator_initialization()
    yield from batch_data_generator(BATCH_SIZE, word_index, 2)

def model_graph(center_word, ctx_word):
    # Embedding matrix for vector lookup
    with tf.name_scope('network'):
        embed_matrix = tf.get_variable(
            shape = [VOCAB_SIZE, EMBED_SIZE],
            initializer = tf.random_uniform_initializer(),
            name = 'embedding'
        )
        # A lookup functions, just returns the corrosponding row
        embed_lookup = tf.nn.embedding_lookup(embed_matrix, center_word,
            name = 'lookup')
        # Reconstruction matrices
        reconstruction_matrix = tf.get_variable(
            shape = [VOCAB_SIZE, EMBED_SIZE],
            initializer = tf.truncated_normal_initializer(stddev=1.0/(EMBED_SIZE**0.5)),
            name = 'reconstruction'
        )
        bias = tf.get_variable(initializer = tf.zeros([VOCAB_SIZE]), name = 'bias')
    # Losses, inbuilt
    with tf.name_scope('loss_optimization'):
        nce_loss = tf.nn.nce_loss(
            weights = reconstruction_matrix,
            biases = bias,
            labels = ctx_word,
            inputs = embed_lookup,
            num_sampled = 64,
            num_classes = VOCAB_SIZE
        )
        loss = tf.reduce_mean(nce_loss, name = 'loss')
        optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
    return embed_matrix, embed_lookup, loss, optimizer

def main():
    #word_index = data_generator_initialization()
    #gen_fn = batch_data_generator(BATCH_SIZE, word_index, 2)
    #gen_fn = generator()

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        (tf.int32, tf.int32),
        (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    with tf.name_scope('inputs'):
        iterator = tf_dataset.make_initializable_iterator()
        center, ctx = iterator.get_next()
    embed_matrix, embed_lookup, loss, optimizer = model_graph(center, ctx)
    writer = tf.summary.FileWriter('Dataset/graphs', tf.get_default_graph())
    # Start training
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        # See if training can be continued
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('Dataset/checkpoint/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Checkpoint restored..")
        step_loss = 0.0
        for nr_steps in range(NR_STEPS):
            try:
                loss_val, _ = sess.run([loss, optimizer])
                step_loss += loss_val
                if (nr_steps + 1) % 1000 == 0:
                    print('Step ' + str(nr_steps+1) + ' loss: ' + str(step_loss))
                    step_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        saver.save(sess, 'Dataset/checkpoint/model.ckpt')
    writer.close()

if __name__ == "__main__":
    main()
