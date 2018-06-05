import tensorflow as tf
import numpy as np

def conv(input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def depthconv(input, filter_size, stride, name, padding = 'SAME', dilation = 1, multiplier = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, multiplier])
        biases = tf.get_variable(name = name + '_biases', shape = [input_channels*multiplier])
        depthconv = tf.nn.depthwise_conv2d(input, weights, strides = [1, stride, stride, 1],
                                      padding = padding, rate = [dilation, dilation], name = name)
        bias = tf.reshape(tf.nn.bias_add(depthconv, biases), depthconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def pointconv(input, nr_filters, stride, name):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = 'SAME', name = name)
        bias = tf.reshape(tf.nn.bias_add(pointconv, biases), pointconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def channel_weighted_pooling(weights, channel, name):
    nr_channel = channel.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        pool_weights = tf.split(weights, num_or_size_splits = nr_channel, axis = 3)
        channel_outputs = tf.split(channel, num_or_size_splits = nr_channel, axis = 3)
        prod = []
        for i in range(nr_channel):
            prod.append(tf.multiply(pool_weights[i], channel_outputs[i]))
        output = tf.concat(prod, axis = 3)
        return output

def intermediate_residual(depth_in, point_in, name):
    nr_channels = int(depth_in.get_shape()[-1])
    with tf.name_scope(name) as scope:
        depthconv_inter = depthconv(depth_in, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth1',
                                        dilation = 1, multiplier = 2)
        pointconv_inter = pointconv(point_in, nr_channels*2, stride = 2,
                                        name = name+'_point1')
        tensor_inter = tf.concat([depth_in, point_in], axis = 3)
        conv_inter = conv(tensor_inter, filter_size = 3, nr_filters = 2*nr_channels,
                            stride = 2, name = name+'_conv', padding = 'SAME')
        depth_out_tensor = tf.concat([depthconv_inter, conv_inter], axis = 3)
        depthconv_out = depthconv(depth_out_tensor, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth2',
                                        dilation = 1, multiplier = 1)
        point_out_tensor = tf.concat([pointconv_inter, conv_inter], axis = 3)
        pointconv_out = pointconv(point_out_tensor, nr_channels*2*2, stride = 2,
                                        name = name+'_point2')
        return depthconv_out, pointconv_out


def model(input):
    with tf.name_scope('model') as scope:
        depthconv1_1 = depthconv(input, filter_size = 5, stride = 1,
                        padding = 'SAME', name = 'depth1_1', dilation = 1, multiplier = 4)
        pointconv1_2 = pointconv(input, nr_filters = 12, stride = 1, name = 'point1_1')

        pool1_1 = tf.nn.max_pool(depthconv1_1, ksize=[1,2,2,1], strides = [1,2,2,1],
                        padding = 'VALID', name = 'pool1_1')
        pool1_2 = tf.nn.max_pool(pointconv1_2, ksize=[1,2,2,1], strides = [1,2,2,1],
                        padding = 'VALID', name = 'pool1_2')

        depthconv3_1, pointconv3_2 = intermediate_residual(pool1_1, pool1_2, 'inter_res1')
        depthconv5_1, pointconv5_2 = intermediate_residual(depthconv3_1, pointconv3_2, 'inter_res2')

        ch_pool = channel_weighted_pooling(depthconv5_1, pointconv5_2, 'cw_pooling')
        avg_pool = tf.reduce_mean(ch_pool, [1,2], keepdims = True)
        print(avg_pool.get_shape().as_list())
        #flat = tf.layers.Flatten(avg_pool)
        flat = tf.reshape(avg_pool, [-1, 1 * 1 * 192], name = 'flattening')
        print(flat.get_shape().as_list())
        logits = tf.layers.dense(flat,100, activation=None, use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer,
                bias_initializer=tf.zeros_initializer(),
                name='dense')
        output = tf.argmax(input=logits, axis=1),
        probab = tf.nn.softmax(logits, name="softmax_tensor")
        return output, probab

if __name__ == '__main__':
    pass
    #x = tf.placeholder(tf.float32, [16, 512, 512, 3])
    #out, prob = model(x)
    #print(out.get_shape().as_list())
    #print(prob.get_shape().as_list())
