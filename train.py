import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from utils_cnn import load_dataset
from model import model

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
#index = 1 # change the index to see another picture
#plt.imshow(X_train_orig[:, index].reshape((28, 28)))
#plt.show()
#print ("y = " + str(np.squeeze(Y_train_orig[index])))

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
label_binrizer = LabelBinarizer()
Y_train = label_binrizer.fit_transform(Y_train_orig)
Y_test = label_binrizer.fit_transform(Y_test_orig)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# train the neural network
_, _, parameters = model(X_train, Y_train, X_test, Y_test)