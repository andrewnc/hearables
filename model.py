"""
Convolutional model for hearables challenge.
Purpose: Clean audio by removing background noise.
Method: Supervised learning with a sweep of 250 ms to output 10ms chunks of audio
"""

import numpy as np
import tensorflow as tf



# From unet3_model.py
# define a layer consisting of a convolution plus bias, followed by a relu and batch norm
def conv( x, filter_size=8, stride=2, num_filters=64, is_output=False, name="conv" ):

    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]
    out_channels = num_filters

    with tf.variable_scope( name ):
        with tf.device('/cpu:0'):
            W = tf.get_variable( "W", shape=[filter_width, in_channels, out_channels],
                                 initializer = tf.contrib.layers.variance_scaling_initializer() )
            b = tf.get_variable( "b", shape=[out_channels],
                                 initializer = tf.contrib.layers.variance_scaling_initializer() )

        conv = tf.nn.conv1d( x, W, stride=stride, padding="SAME")
        out = tf.nn.bias_add(conv, b)
        if not is_output:
#            out = tf.contrib.layers.batch_norm( tf.nn.relu(out) )
            out = tf.nn.relu(out)

    return out

def network(input_data):
    
    # assume input data is [1,5120,1]

    conv1a = conv( input_data, filter_size=32, stride=1, num_filters=64, name="conv1a" )
    conv1b = conv( conv1a, filter_size=32, stride=1, num_filters=64, name="conv1b" )
    conv1c = conv( conv1b, filter_size=32, stride=1, num_filters=64, name="conv1c" )
    conv1d = conv( conv1c, filter_size=32, stride=2, num_filters=64, name="conv1d" )
    #print conv1d.get_shape() # shape should be [1,2560,64]

    conv2a = conv( conv1d, filter_size=32, stride=1, num_filters=64, name="conv2a" )
    conv2b = conv( conv2a, filter_size=32, stride=1, num_filters=64, name="conv2b" )
    conv2c = conv( conv2b, filter_size=32, stride=1, num_filters=64, name="conv2c" )
    conv2d = conv( conv2c, filter_size=32, stride=2, num_filters=64, name="conv2d" )
    # shape should be [1,1280,64]

    conv3a = conv( conv2d, filter_size=32, stride=1, num_filters=64, name="conv3a" )
    conv3b = conv( conv3a, filter_size=32, stride=1, num_filters=64, name="conv3b" )
    conv3c = conv( conv3b, filter_size=32, stride=1, num_filters=64, name="conv3c" )
    conv3d = conv( conv3c, filter_size=32, stride=2, num_filters=64, name="conv3d" )
    # shape should be [1,640,64]

    conv4a = conv( conv3d, filter_size=32, stride=1, num_filters=64, name="conv4a" )
    conv4b = conv( conv4a, filter_size=32, stride=1, num_filters=64, name="conv4b" )
    conv4c = conv( conv4b, filter_size=32, stride=1, num_filters=64, name="conv4c" )
    conv4d = conv( conv4c, filter_size=32, stride=2, num_filters=64, name="conv4d" )
    # shape should be [1,320,64]

    conv5a = conv( conv4d, filter_size=32, stride=1, num_filters=64, name="conv5a" )
    conv5b = conv( conv5a, filter_size=32, stride=1, num_filters=64, name="conv5b" )
    conv5c = conv( conv5b, filter_size=32, stride=1, num_filters=64, name="conv5c" )
    conv5d = conv( conv5c, filter_size=32, stride=2, num_filters=64, name="conv5d" )
    # shape should be [1,160,64]

    output = conv( conv5d, filter_size=1, stride=1, num_filters=1, is_output=True, name="outputs" )

    return output

def network2(input_data):
    conv1a = conv( input_data, filter_size=32, stride=1, num_filters=64, name="conv1a" )
    conv1d = conv( conv1a, filter_size=32, stride=1, num_filters=64, name="conv1d" )
    #print conv1d.get_shape() # shape should be [1,2560,64]

    conv2a = conv( conv1d, filter_size=32, stride=1, num_filters=64, name="conv2a" )
    conv2d = conv( conv2a, filter_size=32, stride=1, num_filters=64, name="conv2d" )
    # shape should be [1,1280,64]

    conv3a = conv( conv2d, filter_size=32, stride=1, num_filters=64, name="conv3a" )
    conv3d = conv( conv3a, filter_size=32, stride=1, num_filters=64, name="conv3d" )
    # shape should be [1,640,64]

    conv4a = conv( conv3d, filter_size=32, stride=1, num_filters=64, name="conv4a" )
    conv4d = conv( conv4a, filter_size=32, stride=1, num_filters=64, name="conv4d" )
    # shape should be [1,320,64]

    conv5a = conv( conv4d, filter_size=32, stride=1, num_filters=64, name="conv5a" )
    conv5d = conv( conv5a, filter_size=32, stride=1, num_filters=64, name="conv5d" )
    # shape should be [1,160,64]

    output = conv( conv5d, filter_size=1, stride=1, num_filters=1, is_output=True, name="outputs" )

    return output
    # conv1a = conv(input_data, filter_size = 32, stride=1, num_filters=64,name="conv1a")
    # return conv(conv1a, filter_size=1, stride=1, num_filters=1, is_output=True, name="outputs" )


def network1(input_data):
    
    # assume input data is [1,5120,1]

    conv1a = conv( input_data, filter_size=32, stride=1, num_filters=64, name="conv1a" )
    conv1d = conv( conv1a, filter_size=32, stride=2, num_filters=64, name="conv1d" )
    #print conv1d.get_shape() # shape should be [1,2560,64]

    conv2a = conv( conv1d, filter_size=32, stride=1, num_filters=64, name="conv2a" )
    conv2d = conv( conv2a, filter_size=32, stride=2, num_filters=64, name="conv2d" )
    # shape should be [1,1280,64]

    conv3a = conv( conv2d, filter_size=32, stride=1, num_filters=64, name="conv3a" )
    conv3d = conv( conv3a, filter_size=32, stride=2, num_filters=64, name="conv3d" )
    # shape should be [1,640,64]

    conv4a = conv( conv3d, filter_size=32, stride=1, num_filters=64, name="conv4a" )
    conv4d = conv( conv4a, filter_size=32, stride=2, num_filters=64, name="conv4d" )
    # shape should be [1,320,64]

    conv5a = conv( conv4d, filter_size=32, stride=1, num_filters=64, name="conv5a" )
    conv5d = conv( conv5a, filter_size=32, stride=2, num_filters=64, name="conv5d" )
    # shape should be [1,160,64]

    output = conv( conv5d, filter_size=1, stride=1, num_filters=1, is_output=True, name="outputs" )

    return output