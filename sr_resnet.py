# -*- coding: utf-8 -*-
"""This is the module where the network is declared."""
import tensorflow as tf
import tensorflow.contrib as tf_contrib


weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def fully_conneted(feature, units, use_bias=True, scope='fully_0'):
    """Fully connected layer function."""
    with tf.variable_scope(scope):
        feature = tf.layers.dense(feature, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return feature

def global_avg_pooling(feature):
    """Global average pooling layer function."""
    gap = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
    return gap

def relu(feature):
    """ReLU layer function."""
    return tf.nn.relu(feature)

def batch_norm(feature, is_training=True, scope='batch_norm'):
    """Batch norm layer function."""
    return tf_contrib.layers.batch_norm(feature,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def conv(feature, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    """Convolution layer function."""
    with tf.variable_scope(scope):
        feature = tf.layers.conv2d(inputs=feature, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return feature

def resblock(x_init, channels, is_training=True, use_bias=True,
             downsample=False, scope='resblock') :
    """Residual block."""
    with tf.variable_scope(scope) :

        feature = batch_norm(x_init, is_training, scope='batch_norm_0')
        feature = relu(feature)

        if downsample :
            feature = conv(feature, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2,
                          use_bias=use_bias, scope='conv_init')

        else :
            feature = conv(feature, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        feature = batch_norm(feature, is_training, scope='batch_norm_1')
        feature = relu(feature)
        feature = conv(feature, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

    return feature + x_init

def spatial_attention(input_feature, name):
    """Spatial attention module."""
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat

def temporal_attention(input_feature, keep_prob, name):
    """Temporal attention module."""
    with tf.variable_scope(name):
        residual = input_feature

        avg_feature = tf.reduce_mean(input_feature, axis=[1], keepdims=False)
        concat_feature = []
        for i in range(input_feature.shape[1]):
            concat_feature.append(tf.concat([input_feature[:, i, :], avg_feature], axis=1))
        concat_feature_stack = tf.stack(concat_feature, axis=1)
        concat_feature_reshape = tf.reshape(concat_feature_stack, [-1, 1024])

        temporal_att = tf.nn.dropout(concat_feature_reshape, keep_prob)
        temporal_att = tf.layers.dense(temporal_att, units=1024, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             use_bias=False, activation=tf.nn.relu)
        temporal_att = tf.layers.dense(temporal_att, units=1024, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             use_bias=False, activation=tf.nn.relu)
        temporal_att = tf.layers.dense(temporal_att, units=1, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             use_bias=False, activation=tf.nn.sigmoid)

        ta_reshape = tf.reshape(temporal_att, [-1, 48, 1])
        ta_x = input_feature * ta_reshape
        out = residual + ta_x

    return input_feature * out

def network(feature_maps, keep_prob=1.0, is_training=True, reuse=tf.AUTO_REUSE):
    """Residual network module."""
    with tf.variable_scope("retrain_net", reuse=reuse):
        feature = resblock(feature_maps, channels=512, is_training=is_training, downsample=True,
                            scope='resblock_3_0')
        feature = resblock(feature, channels=512, is_training=is_training, downsample=False,
                            scope='resblock_3_1')
        feature = resblock(feature, channels=512, is_training=is_training, downsample=False,
                            scope='resblock_3_2')

        feature = batch_norm(feature, is_training, scope='batch_norm')
        feature = relu(feature)

        residual = feature
        feature = spatial_attention(feature, name='sa') * feature
        feature += residual
        feature = relu(feature)

        feature = global_avg_pooling(feature)

        feature = tf.reshape(feature, [-1, 48, 512])
        feature = temporal_attention(feature, keep_prob=keep_prob, name='ta')

        final_feature = tf.reduce_mean(feature, axis=[1], keepdims=False)
        logits = fully_conneted(final_feature, 3)

    return logits, final_feature

def cross_entropy_loss(logit, label):
    """Cross entropy loss function."""
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,
                                                                     logits=logit))

    return loss

def calculate_accuracy(logit, label, name) :
    """Accuracy calculation function."""
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name=name)

    return accuracy
