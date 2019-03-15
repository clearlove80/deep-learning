import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from mnist_model import mnist_inference_fnn
from tensorflow.contrib import layers

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = ''
MODEL_NAME = ''


def train(mnist):
    x = tf.placeholder(
        tf.float32, [
            None,
            mnist_inference_fnn.INPUT_NODE], name='input-x')
    t = tf.placeholder(
        tf.float32, [
            None,
            mnist_inference_fnn.OUTPUT_NODE], name='input-t')

    regularizer = layers.l2_regularizer(REGULARAZATION_RATE)
    y = mnist_inference_fnn.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    var_avg = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    var_avg_op = var_avg.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(t, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        TRAINING_STEPS,
        mnist.train.num_examples /
        BATCH_SIZE,
        LEARNING_RATE_DECAY)
    #train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_op = tf.group(train_step, var_avg_op)

    correct_prediction = tf.equal(tf.argmax(t, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, TRAINING_STEPS + 1):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # xs = np.reshape(xs,
            #                 [BATCH_SIZE,
            #                  mnist_inference_LeNet_5.IMAGE_SIZE,
            #                  mnist_inference_LeNet_5.IMAGE_SIZE,
            #                  mnist_inference_LeNet_5.NUM_CHANNELS])
            # print(ys)
            _, loss_value = sess.run(
                [train_op, loss], feed_dict={x: xs, t: ys})
            # if i % 1000 == 0:
            #     print(
            #         'After %d training step(s), loss on training batch is %g.' %
            #         (i, loss_value))

            if i % 1000 == 0:
                xs = mnist.validation.images
                # xs = np.reshape(xs,
                #                 [mnist.validation.num_examples,
                #                  mnist_inference_LeNet_5.IMAGE_SIZE,
                #                  mnist_inference_LeNet_5.IMAGE_SIZE,
                #                  mnist_inference_LeNet_5.NUM_CHANNELS])
                ys = mnist.validation.labels
                accuracy_validation = sess.run(
                    accuracy, feed_dict={x: xs, t: ys})
                print(
                    'After %d training step(s), loss on training batch is %g.' %
                    (i, loss_value), 'Accuracy on validaton is %g.' %
                    accuracy_validation, sep='   ')
        xs = mnist.test.images
        # xs = np.reshape(xs,
        #                 [mnist.test.num_examples,
        #                  mnist_inference_LeNet_5.IMAGE_SIZE,
        #                  mnist_inference_LeNet_5.IMAGE_SIZE,
        #                  mnist_inference_LeNet_5.NUM_CHANNELS])
        ys = mnist.test.labels
        print(
            'Accuracy on test is %g.' %
            sess.run(
                accuracy,
                feed_dict={
                    x: xs,
                    t: ys}))

        while True:
            index = int(input('input one number:'))
            image = mnist.test.images[index]
            image = np.reshape(image, [1, mnist_inference_fnn.INPUT_NODE])
            label = tf.argmax(mnist.test.labels[index], -1)
            prediction=tf.argmax(tf.nn.softmax(y),1)
            prediction,label = sess.run([prediction,label], feed_dict={x: image})
            print('prediction:', prediction)
            print('fact:', label)
            print('image:')
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap=plt.get_cmap('gray_r'))
            plt.show()




def main(argv=None):
    mnist = input_data.read_data_sets(
        'D:/Deep_Learning/MNIST_DATASET', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
