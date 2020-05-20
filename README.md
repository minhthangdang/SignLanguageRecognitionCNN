# Sign Language Recognition using CNN
Convolution Neural Network for sign language recognition using MNIST image dataset

## Problem Statement

This is my own exercise resulted from Coursera's Convolutional Neural Networks.

This is my second deep learning model for sign language recognition using MNIST
image dataset. This model differs from the first one (https://github.com/minhthangdang/SignLanguageRecognition) in that this one uses Convolutional Neural Network.

## The Dataset

The dataset is obtained from Kaggle (https://www.kaggle.com/datamunge/sign-language-mnist).
The training data has 27,455 examples and the test data has 7,172 examples. Each
example is a 784 (28x28) pixel vector with grayscale values between 0-255. It has
24 classes of letters (excluding J and Z) in American Sign Language.

An illustration of the sign language is shown here (image courtesy of Kaggle):

<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/datasets_3258_5337_amer_sign2.png" alt="Sign Language" width="400"/><br>

Grayscale images with (0-255) pixel values:

<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/datasets_3258_5337_amer_sign3.png" alt="Sign Language" width="400"/><br>

One example in the MNIST dataset:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/mnist-example.JPG?raw=true" alt="Sign Language" width="400"/><br>

## Convolutional Neural Network Architecture

My network architecture borrowed the ideas of LeNet-5 model (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), thanks to its relatively simple and easy to train network. The architecture is as follows:

CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> FULLYCONNECTED -> FULLYCONNECTED

The architecture is depicted below:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/conv-net-sign.JPG?raw=true" alt="Convolutional Network Architecture"/><br>

The hyperparameter values are learning_rate = 0.0001, num_epochs = 30, minibatch_size = 64,
and optimizer = AdamOptimizer.

The program is written in Python and Tensorflow 1.x

The result is:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/sign-cost-cnn.JPG?raw=true" alt="Cost Function Plot" width="400"/><br>

```
Train Accuracy: 1.0
Test Accuracy: 0.89445066
```

Even though it is a relatively simple network, it achieved very good results. Compared to my first deep learning model for the same task (https://github.com/minhthangdang/SignLanguageRecognition), this model is not only better in accuracy, but also faster in training time with much lower number of epochs.

This is one of my repositories in a series of deep learning exercises. Please check out
my other repositories to see more.

Should you have any questions, please contact me via Linkedin: https://www.linkedin.com/in/minh-thang-dang/
