
# Homework 1, [Columbia EECS 6894](https://columbia6894.github.io/)

Note: all the programming tasks must be finished with [Jupyter notebook](http://jupyter.org/)


## Problem 1
Tensorflow can compute the gradient automatically. The following code snippet shows an example of
computing the gradient for mean square loss.

```py
import tensorflow as tf
label = tf.constant(1.0, dtype=tf.float32)
x = tf.placeholder(tf.float32)

loss_mse = tf.losses.mean_squared_error(label, x)
gradient_mse = tf.gradients(loss_mse, x)

with tf.Session() as sess:
    ci, gi = sess.run((loss_mse, gradient_mse), feed_dict={x: 1.0})
    print 'mse, loss, grad = ', ci, gi

```
Please extend the above code to compute:
1. The gradient of hinge loss when label = 1.0, x = 1.001
2. The gradient of hinge loss when label = 1.0, x = 0.009
3. Plot the curves of gradients and losses for x in [-2, 2]  (hint: use `%matplotlib inline` in Jypiter)

## Problem 2
Logistic regression and multi-layer perceptrons (MLPs) are two basic models for classification tasks.
Try to use these two models to learn from the following xor dataset:

```
import numpy
xs = np.array([[-1.1, 1.0], [-1.0, 1.1], [-1.1, 1.1], [1.0, -1.1],[1.1, -1.0],[1.0, -1.0],
                  [1.1, 1.1],[1.0, 0.9],[1.1, 1.0],  [-1.1, -1.0], [-1.1, -1.1], [-1.0, -1.1]],
                dtype=np.float32)
ys = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
ys = ys[:, None]
```
Train and Evaluate your models using `xs` (samples) and `ys` (training labels). You are allowed to use
[Keras](https://keras.io/). Compare the performance of your two models.


## Problem 3
[The MNIST dataset of handwritten digits](http://yann.lecun.com/exdb/mnist/) is a very popular dataset to
test the algorithms and ideas of machine learning. To train MNIST data, the following procedures are adopted:

- Reshape the digit pictures ( each with 28x28 pixels) to vectors of 784
- Change the type of xs to float32
- Every pixel is from 0 to 255. Renormalize it to 0 and 1
- Reshape the label vectors ys if necessary

The original MNIST data has 10 categories. Our new task is to take only two categories: digit 4 and digit 8 and
train a classifier. You are suggested to compare two models:
1. One hidden layer MLP with cross entropy loss
2. One hidden layer MLP with hinge loss
3. (bonus) MLP with two and three hidden layers
*hint: you may refer to [Keras example](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)*.


