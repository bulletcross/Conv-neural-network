import tensorflow as tf
import numpy as np
import os
import classification_model as m

lr_rate = 0.00001
batch_size = 16
nr_epochs = 10

def main():
    logs_path = os.path.join(os.getcwd(), 'tf_log')
    print("Printing logs and graphs into: " + logs_path)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 100])
    result, prob = m.model(x)
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data(label_mode = 'fine')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/225
    x_test = x_test/225

    print(x_train.shape)
    print(y_train.shape)
    nr_step = int(x_train.shape[0]/batch_size)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = prob))
    tf.summary.scalar('loss', loss)
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(result)
    tf.summary.scalar('accuracy', accuracy)
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(lr_rate).minimize(loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    ######################################################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                small_feed_x = x_train[step*batch_size: (step+1)*batch_size]
                feed_y = y_train[step*batch_size: (step+1)*batch_size]
                feed_x_op = tf.image.resize_images(small_feed_x,[512, 512])
                feed_x = np.array(sess.run([feed_x_op]))
                feed_x = np.squeeze(feed_x, axis=0)
                #feed_x = feed_x.reshape([batch_size, 512, 512, 3])
                _, step_loss, step_accuracy = sess.run([train_op,loss,accuracy], feed_dict = {x: feed_x, y:feed_y})
                if step%100 == 0:
                    summary = sess.run(merged_summary, feed_dict = {x: feed_x, y:feed_y})
                    writer.add_summary(summary, epoch*batch_size+step)
                    print('Epoch= %d, step= %d,loss= %.4f, accuracy= %.4f' % (epoch, step, step_loss, step_accuracy))
        chk_name = os.path.join(logs_path, 'model.ckpt')
        save_path = saver.save(sess, chk_name)

if __name__ == '__main__':
    main()
