import tensorflow as tf

INPUT_NODE = 28**2
OUTPUT_NODE = 10
LAY1NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        'weights',
        shape,
        initializer=tf.truncated_normal_initializer(
            stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAY1NODE], regularizer)
        bias = tf.get_variable(
            'bias',
            [LAY1NODE],
            initializer=tf.constant_initializer(0.))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

    with tf.variable_scope('layer2'):
        weights = tf.get_variable(
            'weights', [
                LAY1NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(
                stddev=0.1))
        bias = tf.get_variable(
            'bias',
            [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.))
        layer2 = tf.matmul(layer1, weights) + bias

    return layer2
